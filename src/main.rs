use async_openai::{types::CreateEmbeddingRequestArgs, Client};
use clap::{Parser, Subcommand};
use polars::{lazy::dsl::GetOutput, prelude::*};
use std::path::Path;

const MAX_TOKEN: usize = 8100;
const CHUNK_SIZE: usize = 256;
const TAGS: &str = "<quantum-mechanics>|<statistical-mechanics>|<thermodynamics>|<electromagnetism>|<electrodynamics>";

#[derive(Parser)]
#[command(version, about, long_about = None)]
#[command(propagate_version = true)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Parse and cleanup the XML into Parquet
    // This should update the parquet file in place incrementally
    ParseXML { input: String, output: String },
    /// Generate the embedding for the given Parquet input
    // This should update the parquet incrementally
    Brew { input: String, output: String },
    /// Vector search the database with text
    Search { input: String, text: String },
}

fn main() {
    let cli = Cli::parse();

    if let Err(e) = match cli.command {
        Commands::ParseXML { input, output } => parse_xml(input, output),
        Commands::Brew { input, output } => brew(input, output),
        Commands::Search { input, text } => search(input, text),
    } {
        println!("{}", e);
    }
}

#[tokio::main]
async fn search(input: String, text: String) -> PolarsResult<()> {
    std::env::set_var("POLARS_FMT_MAX_ROWS", "20");
    std::env::set_var("POLARS_FMT_STR_LEN", "50");

    let text_embedding = Series::new("embedding", get_embedding(text).await.unwrap());

    let df = LazyFrame::scan_parquet(input, Default::default())?
        .with_columns([
            (lit("https://physics.stackexchange.com/questions/")
                + col("id").cast(DataType::String))
            .alias("id"),
            col("embeddings")
                .map(
                    move |c| {
                        Ok(Some(ChunkedArray::<Float64Type>::into_series(
                            c.list()?
                                .apply_nonnull_values_generic(DataType::Float64, |e| {
                                    Series::from_arrow("embedding", e)
                                        .unwrap()
                                        .dot(&text_embedding)
                                        .unwrap()
                                }),
                        )))
                    },
                    GetOutput::from_type(DataType::Float64),
                )
                .alias("score"),
        ])
        .sort(
            "score",
            SortOptions {
                descending: true,
                nulls_last: true,
                ..Default::default()
            },
        )
        .select([cols(["id", "title", "score"])])
        .collect()?;

    println!("{}", df.head(Some(20)));

    Ok(())
}

// FIXME: Could use lazy_static etc.
async fn get_embedding(
    //    client: &Client<OpenAIConfig>,
    //    tokenizer: &CoreBPE,
    input: String,
) -> Option<Vec<f32>> {
    let client = Client::new();
    let tokenizer = tiktoken_rs::cl100k_base().unwrap();

    let token_len = tokenizer.encode_ordinary(&input).len();
    if token_len > MAX_TOKEN {
        println!("Token too long, len: {}, prompt: {}", token_len, input);
        return None;
    }

    let req = CreateEmbeddingRequestArgs::default()
        .model("text-embedding-3-large")
        .input(input)
        .build()
        .ok()?;

    Some(
        client
            .embeddings()
            .create(req)
            .await
            .map_err(|x| dbg!(x))
            .ok()?
            .data
            .pop()?
            .embedding,
    )
}

#[tokio::main]
async fn get_embeddings(series: &mut [Series]) -> PolarsResult<Option<Series>> {
    use itertools::Itertools;

    let mut results: Vec<Option<Series>> = Vec::new();

    let zipped = series[0].str()?.iter().zip(
        series[1]
            .bool()?
            .iter()
            .map(|x| x.expect("mask must be non-null")),
    );

    for xs in zipped.chunks(CHUNK_SIZE).into_iter() {
        let handles: Vec<_> = xs
            .map(|(text, mask)| {
                if mask {
                    Some(tokio::spawn(get_embedding(text.unwrap().to_string())))
                } else {
                    None
                }
            })
            .collect();
        for handle in handles {
            results.push(if let Some(handle) = handle {
                handle.await.unwrap().map(|x| Series::new("embedding", x))
            } else {
                None
            });
        }
    }

    Ok(Some(
        ChunkedArray::<ListType>::from_iter(results.into_iter()).into_series(),
    ))
}

// Currently, you have to modify the code here to filter what you want to brew
fn brew(input: String, output: String) -> PolarsResult<()> {
    let filtering = col("tags")
        .str()
        .contains(lit(TAGS), false)
        .and(col("embeddings").is_null());

    // The when-then-otherwise is not lazy, so we need to manually return if filtering indicates no update is needed
    if LazyFrame::scan_parquet(&input, Default::default())?
        .filter(filtering.clone())
        .collect()?
        .height()
        == 0
    {
        println!("No update needed");
        return Ok(());
    }

    let mut df = LazyFrame::scan_parquet(input, Default::default())?
        .with_columns([
            (lit("Title: ") + col("title") + lit(" Body: ") + col("body")).alias("combined"),
        ])
        .with_column(filtering.alias("mask"))
        .with_column(
            // NOTE: If we create filter such that there is no update then we will get an error on not being able to convert the return type.
            map_multiple(
                get_embeddings,
                &[col("combined"), col("mask")],
                GetOutput::from_type(DataType::List(DataType::Float32.boxed())),
            )
            .alias("masked_updates"),
        )
        // This by default updates the "embeddings" column
        .with_column(coalesce(&[col("embeddings"), col("masked_updates")]))
        .select([cols(["id", "title", "body", "tags", "embeddings"])])
        .collect()?;

    println!("{}", df);

    let mut file = std::fs::File::create(output).unwrap();
    // Use the default zstd compression
    ParquetWriter::new(&mut file).finish(&mut df)?;

    println!("Finished writing");

    Ok(())
}

fn parse_xml(input: impl AsRef<Path>, output: impl AsRef<Path>) -> PolarsResult<()> {
    use roxmltree::Document;

    let mut df = DataFrame::default();
    let text = std::fs::read_to_string(input).unwrap();

    for node in Document::parse(&text).unwrap().descendants() {
        // Make sure we have got a valid question post
        if node
            .attributes()
            .any(|a| (a.name() == "PostTypeId") && (a.value() == "1"))
            && node
                .attributes()
                .any(|a| (a.name() == "Score") && (a.value().parse::<isize>().unwrap() >= 0))
        {
            // By definition these attributes must exists as we have got a question already
            let id: u32 = node
                .attributes()
                .find(|a| a.name() == "Id")
                .expect("Question Post expects Id")
                .value()
                .parse()
                .expect("Question Id should be u32");
            // Remove HTML tags and trim the text
            let body = voca_rs::strip::strip_tags(
                node.attributes()
                    .find(|a| a.name() == "Body")
                    .expect("Question Post expects Body")
                    .value()
                    .trim(),
            );
            let tags = node
                .attributes()
                .find(|a| a.name() == "Tags")
                .expect("Question Post expects Tags")
                .value();
            let title = node
                .attributes()
                .find(|a| a.name() == "Title")
                .expect("Question Post expects Title")
                .value();

            let row = df!("id" => &[id], "title" => &[title], "body" => &[body], "tags" => &[tags], "embeddings" => &[None::<Series>])?;
            df.vstack_mut(&row)?;
        }
    }
    println!("{}", df);

    let mut file = std::fs::File::create(output).unwrap();
    // Use the default zstd compression
    ParquetWriter::new(&mut file).finish(&mut df)?;

    println!("Finished writing");

    Ok(())
}
