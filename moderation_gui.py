import gradio as gr
import pandas as pd
from openai import AsyncOpenAI
import os
import tiktoken


aclient = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def count_token(input_text):
    enc = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(enc.encode(input_text))
    return num_tokens


async def call_openai_api(input_text):
    response = await aclient.moderations.create(input=input_text)
    return response.dict()


def convert_json_to_df(data):
    categories = data["results"][0]["categories"]
    category_scores = data["results"][0]["category_scores"]

    df_categories = pd.DataFrame(
        list(categories.items()), columns=["category", "flagged"]
    )
    df_category_scores = pd.DataFrame(
        list(category_scores.items()), columns=["category", "score"]
    )
    df_category_scores["score"] = df_category_scores["score"].astype(float)

    df_combined = pd.merge(df_categories, df_category_scores, on="category")
    df_info = pd.DataFrame(
        {
            "id": [data["id"]],
            "model": [data["model"]],
            "flagged": [data["results"][0]["flagged"]],
        }
    )

    return df_info, df_combined


async def main_func(input_text):
    data = await call_openai_api(input_text)
    dfs = convert_json_to_df(data)
    return dfs + (len(input_text), count_token(input_text))


iface = gr.Interface(
    fn=main_func,
    inputs=[
        gr.Textbox(
            label="モデレーション対象",
            lines=2,
            placeholder="ここにテキストを入力してください",
            show_copy_button=True,
        ),
    ],
    outputs=[
        gr.Dataframe(label="Info", headers=["id", "model", "flagged"]),
        gr.Dataframe(label="結果", headers=["category", "flagged", "score"]),
        gr.Number(label="文字数"),
        gr.Number(label="トークン数"),
    ],
    title="テキストモデレーションツール",
    description="テキストを入力して、OpenAIのモデレーション結果を取得します。",
)

iface.launch(share=False, debug=True)
