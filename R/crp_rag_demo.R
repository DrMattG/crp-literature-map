# CRP Abstract RAG Demo -------------------------------------------------------
#
# Purpose:
# This script demonstrates how to build a small retrieval-augmented generation
# (RAG) workflow over OpenAlex title and abstract records using ragnar and ellmer.
#
# Important limitation:
# The chatbot retrieves and answers from titles and abstracts only. It should be
# treated as a search, mapping, and scoping assistant rather than a tool for
# drawing definitive conclusions from full-text evidence.

# Packages -------------------------------------------------------------------

library(jsonlite)
library(dplyr)
library(stringr)
library(ragnar)
library(ellmer)

# Helper functions ------------------------------------------------------------

#' Clean text for embedding and retrieval
#'
#' Converts text to UTF-8, removes problematic control/non-printing characters,
#' and collapses repeated whitespace. This is useful when records come from large
#' bibliographic JSON files where abstracts may include OCR artefacts or unusual
#' Unicode characters.
#'
#' @param x A character vector, or an object that can be coerced to character.
#'
#' @return A cleaned character vector.
#'
#' @examples
#' clean_text("Th is text has odd spacing\n\nand control characters")
clean_text <- function(x) {
  x |>
    as.character() |>
    iconv(from = "", to = "UTF-8", sub = " ") |>
    str_replace_all("[^[:print:]\n\r\t]", " ") |>
    str_replace_all("[[:cntrl:]]", " ") |>
    str_squish()
}

#' Prepare OpenAlex records for RAG
#'
#' Takes an OpenAlex-style records data frame and creates one retrievable text
#' unit per article. Each unit contains the title and abstract, with metadata
#' retained for citation and inspection.
#'
#' @param records A data frame containing OpenAlex records. Expected columns are
#'   `id`, `doi`, `title`, `abstract`, `publication_year`, and `journal`.
#' @param max_chars Maximum number of characters to keep in the text sent for
#'   embedding. Truncation helps avoid over-large embedding requests.
#'
#' @return A tibble with cleaned metadata and a `text` column for embedding.
#'
#' @examples
#' # chunks <- prepare_openalex_chunks(records, max_chars = 6000)
prepare_openalex_chunks <- function(records, max_chars = 6000) {
  records |>
    transmute(
      id = as.character(id),
      doi = if_else(is.na(doi), "", as.character(doi)),
      title = clean_text(title),
      year = as.integer(publication_year),
      journal = if_else(is.na(journal), "", clean_text(journal)),
      abstract = clean_text(abstract),
      text = str_glue("{title}\n\n{abstract}")
    ) |>
    filter(
      !is.na(title),
      !is.na(abstract),
      abstract != "",
      !is.na(text),
      text != "",
      nchar(text) > 20
    ) |>
    mutate(text = str_sub(text, 1, max_chars))
}

#' Create and populate a ragnar store in batches
#'
#' Creates a DuckDB-backed ragnar store, inserts chunks in small batches, and
#' builds the retrieval index. Batching helps avoid HTTP 400 errors from sending
#' too much text to the embedding endpoint in a single request.
#'
#' @param chunks A data frame produced by `prepare_openalex_chunks()`.
#' @param path Path to the DuckDB file to create.
#' @param model OpenAI embedding model name.
#' @param batch_size Number of rows to insert per batch.
#' @param overwrite Logical. Should an existing store at `path` be overwritten?
#'
#' @return A populated ragnar store object.
#'
#' @examples
#' # store <- create_rag_store(chunks, "crp_abstracts.duckdb")
create_rag_store <- function(
  chunks,
  path = "crp_abstracts.duckdb",
  model = "text-embedding-3-small",
  batch_size = 50,
  overwrite = TRUE
) {
  store <- ragnar_store_create(
    path,
    embed = \(x) embed_openai(x, model = model),
    extra_cols = vctrs::vec_ptype(chunks),
    version = 1,
    overwrite = overwrite
  )

  for (i in seq(1, nrow(chunks), by = batch_size)) {
    j <- min(i + batch_size - 1, nrow(chunks))
    message("Inserting rows ", i, " to ", j)
    ragnar_store_insert(store, chunks[i:j, ])
  }

  ragnar_store_build_index(store)
  store
}

#' Retrieve matching abstracts for a question
#'
#' Convenience wrapper around `ragnar_retrieve()` that returns readable metadata
#' and retrieved text.
#'
#' @param store A populated ragnar store.
#' @param query A search question or topic.
#' @param top_k Number of records to retrieve.
#'
#' @return A tibble containing retrieved titles, years, journals, DOIs, and text.
#'
#' @examples
#' # retrieve_abstracts(store, "nitrate losses", top_k = 5)
retrieve_abstracts <- function(store, query, top_k = 10) {
  ragnar_retrieve(store, query, top_k = top_k) |>
    select(title, year, journal, doi, text)
}

#' Create a cautious RAG chat assistant
#'
#' Creates an ellmer chat object and registers the ragnar retrieval tool. The
#' system prompt makes clear that answers are based on titles and abstracts, not
#' full texts.
#'
#' @param store A populated ragnar store.
#' @param model OpenAI chat model name.
#'
#' @return An ellmer chat object with the ragnar retrieval tool registered.
#'
#' @examples
#' # chat <- create_abstract_chat(store)
#' # chat$chat("What are the main themes?")
create_abstract_chat <- function(store, model = "gpt-4.1") {
  chat <- chat_openai(
    model = model,
    system_prompt = paste(
      "You answer questions using retrieved titles and abstracts.",
      "Be cautious: these are abstracts, not full texts.",
      "Cite the title, year, journal and DOI where possible.",
      "If the retrieved evidence is weak or irrelevant, say so."
    )
  )

  ragnar_register_tool_retrieve(chat, store)
  chat
}

# Workflow --------------------------------------------------------------------

# 1. Load the OpenAlex JSON file.
raw <- fromJSON("crp_openalex_enhanced.json")
records <- raw$records

# 2. Prepare one retrievable text unit per record.
chunks_rag <- prepare_openalex_chunks(records, max_chars = 6000)

# 3. Create the vector store. This step calls the embedding API.
store <- create_rag_store(
  chunks = chunks_rag,
  path = "crp_abstracts.duckdb",
  model = "text-embedding-3-small",
  batch_size = 50,
  overwrite = TRUE
)

# 4. Test retrieval before using a chatbot.
retrieve_abstracts(
  store,
  "What evidence is there that the Conservation Reserve Program reduces nitrate losses?",
  top_k = 10
)

# 5. Create the RAG chat assistant.
chat <- create_abstract_chat(store, model = "gpt-4.1")

# 6. Ask a synthesis-style question.
chat$chat(
  "What are the main themes in this literature on the Conservation Reserve Program?"
)

# 7. Try a small set of example evidence-map queries.
queries <- c(
  "biodiversity benefits of conservation reserve program",
  "water quality nitrate runoff tile drainage",
  "land use change slippage effects",
  "economic evaluation conservation reserve program"
)

retrieval_examples <- lapply(queries, \(q) {
  ragnar_retrieve(store, q, top_k = 5) |>
    select(title, year, doi)
})

names(retrieval_examples) <- queries
retrieval_examples
