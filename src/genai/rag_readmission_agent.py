from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    # Preferred package for Ollama integration in recent LangChain versions.
    from langchain_ollama import OllamaLLM
except ImportError:  # pragma: no cover
    try:
        # Backward-compatible import for older setups.
        from langchain_community.llms import Ollama as OllamaLLM
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Ollama LangChain integration is missing. Install `langchain-ollama` "
            "or `langchain-community`."
        ) from exc


@dataclass
class RetrievalChunk:
    text: str
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple LangChain RAG agent for readmission analytics")
    parser.add_argument("--question", default=None, help="Single question. If omitted, interactive mode starts.")
    parser.add_argument("--model", default="qwen2:7b", help="Ollama model name")
    parser.add_argument("--top-k", type=int, default=8, help="Number of retrieved chunks")
    parser.add_argument("--data-final-dir", default="Data/final", help="Path to final data directory")
    return parser.parse_args()


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _safe_read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_knowledge_chunks(final_dir: Path) -> list[RetrievalChunk]:
    chunks: list[RetrievalChunk] = []

    reason_df = _safe_read_csv(final_dir / "readmission_reason_summary.csv")
    if not reason_df.empty:
        for _, row in reason_df.head(300).iterrows():
            chunks.append(
                RetrievalChunk(
                    text=(
                        f"Condition={row.get('Condition')} total_admissions={row.get('total_admissions')} "
                        f"readmissions_under_30_days={row.get('readmissions_under_30_days')} "
                        f"readmission_30d_rate_pct={row.get('readmission_30d_rate_pct')} "
                        f"avg_lace_score={row.get('avg_lace_score')} avg_los={row.get('avg_los')}"
                    ),
                    source="readmission_reason_summary.csv",
                )
            )

    driver_df = _safe_read_csv(final_dir / "readmission_driver_summary.csv")
    if not driver_df.empty:
        for _, row in driver_df.iterrows():
            chunks.append(
                RetrievalChunk(
                    text=(
                        f"lace_band={row.get('lace_band')} patients={row.get('patients')} "
                        f"readmission_30d_count={row.get('readmission_30d_count')} "
                        f"readmission_30d_rate_pct={row.get('readmission_30d_rate_pct')} "
                        f"avg_los={row.get('avg_los')} avg_chronic_conditions={row.get('avg_chronic_conditions')}"
                    ),
                    source="readmission_driver_summary.csv",
                )
            )

    feature_df = _safe_read_csv(final_dir / "readmission_feature_importance.csv")
    if not feature_df.empty:
        for _, row in feature_df.head(80).iterrows():
            chunks.append(
                RetrievalChunk(
                    text=f"model_feature={row.get('feature')} importance={row.get('importance')}",
                    source="readmission_feature_importance.csv",
                )
            )

    metrics_payload = _safe_read_json(final_dir / "readmission_model_metrics.json")
    if metrics_payload:
        metrics = metrics_payload.get("metrics", {})
        cm = metrics.get("confusion_matrix", {})
        chunks.append(
            RetrievalChunk(
                text=(
                    f"model_metrics roc_auc={metrics.get('roc_auc')} pr_auc={metrics.get('pr_auc')} "
                    f"accuracy={metrics.get('accuracy')} precision={metrics.get('precision')} "
                    f"recall={metrics.get('recall')} f1={metrics.get('f1')} selected_model={metrics.get('selected_model')} "
                    f"tn={cm.get('tn')} fp={cm.get('fp')} fn={cm.get('fn')} tp={cm.get('tp')}"
                ),
                source="readmission_model_metrics.json",
            )
        )

    model_df = _safe_read_csv(final_dir / "readmission_model_dataset.csv")
    if not model_df.empty:
        target_col = "readmitted_under_30_days"
        if target_col in model_df.columns:
            pos_rate = pd.to_numeric(model_df[target_col], errors="coerce").fillna(0).mean() * 100.0
            chunks.append(
                RetrievalChunk(
                    text=f"dataset_rows={len(model_df)} target_positive_rate_pct={pos_rate:.2f}",
                    source="readmission_model_dataset.csv",
                )
            )

        if {"Condition", "readmitted_under_30_days"}.issubset(model_df.columns):
            condition_rates = (
                model_df.groupby("Condition", dropna=False)["readmitted_under_30_days"]
                .mean()
                .mul(100)
                .sort_values(ascending=False)
                .head(40)
                .reset_index(name="readmission_30d_rate_pct")
            )
            for _, row in condition_rates.iterrows():
                chunks.append(
                    RetrievalChunk(
                        text=(
                            f"condition_rate Condition={row.get('Condition')} "
                            f"readmission_30d_rate_pct={row.get('readmission_30d_rate_pct'):.2f}"
                        ),
                        source="readmission_model_dataset.csv",
                    )
                )

    if not chunks:
        chunks.append(
            RetrievalChunk(
                text="No analytics files were found in Data/final.",
                source="system",
            )
        )

    return chunks


def retrieve_context(chunks: list[RetrievalChunk], question: str, top_k: int) -> list[RetrievalChunk]:
    corpus = [chunk.text for chunk in chunks]

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=20000)
    matrix = vectorizer.fit_transform(corpus)
    q_vec = vectorizer.transform([question])

    scores = (matrix @ q_vec.T).toarray().ravel()
    if not np.any(scores):
        return chunks[:top_k]

    top_indices = np.argsort(-scores)[:top_k]
    return [chunks[i] for i in top_indices]


def build_rag_chain(model_name: str):
    prompt = PromptTemplate.from_template(
        """
You are a healthcare readmission analytics assistant.
Use ONLY the provided context.
If the answer is not in context, say that clearly.
Always cite evidence IDs like [1], [2].

Question:
{question}

Context:
{context}

Answer (short bullet points):
""".strip()
    )

    llm = OllamaLLM(model=model_name, temperature=0)
    chain = prompt | llm | StrOutputParser()
    return chain


def answer_question(question: str, chunks: list[RetrievalChunk], model_name: str, top_k: int) -> str:
    retrieved = retrieve_context(chunks, question, top_k=top_k)
    context_lines = [f"[{idx}] source={c.source} | {c.text}" for idx, c in enumerate(retrieved, start=1)]
    context = "\n".join(context_lines)

    chain = build_rag_chain(model_name)
    answer = chain.invoke({"question": question, "context": context})

    if not str(answer).strip():
        return "No model answer returned. Retrieved evidence:\n" + context

    return str(answer)


def run_interactive(chunks: list[RetrievalChunk], model_name: str, top_k: int) -> None:
    print("Simple LangChain RAG agent ready. Type 'exit' to quit.")

    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("Bye.")
            return
        if not question:
            continue

        result = answer_question(question, chunks, model_name=model_name, top_k=top_k)
        print(f"\nAgent:\n{result}")


def main() -> None:
    args = parse_args()
    final_dir = Path(args.data_final_dir)
    chunks = build_knowledge_chunks(final_dir)

    if args.question:
        print(answer_question(args.question, chunks, model_name=args.model, top_k=args.top_k))
        return

    run_interactive(chunks, model_name=args.model, top_k=args.top_k)


if __name__ == "__main__":
    main()
