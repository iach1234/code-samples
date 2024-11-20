import json
import os

import requests
import cohere
import google.generativeai as genai
import backoff
import asyncio
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/search_hits")
def search_hits(request: dict = Body(...)) -> list[dict]:
    query = request.get("query", "")

    cohere_api_key = os.getenv("COHERE_API_KEY")
    co = cohere.Client(cohere_api_key)

    def cohere_rr_response(docs: list[dict]):
        rank_fields = ["text"]

        response = co.rerank(
            model="rerank-multilingual-v3.0",
            query=query,
            documents=docs,
            rank_fields=rank_fields,
            top_n=len(docs),
            return_documents=True,
        )
        return response

    openai_embedding_endpoint = os.getenv("OPENAI_EMBEDDING_ENDPOINT")
    openai_embedding_api = os.getenv("OPENAI_EMBEDDING_API")
    headers = {
        "Content-Type": "application/json",
        "api-key": openai_embedding_api,
    }
    data = {"input": query}

    response = requests.post(openai_embedding_endpoint, headers=headers, json=data)

    tes = response.json()

    tensor_tes = tes["data"][0]["embedding"]

    url = "http://localhost:8080/search/"

    headers = {"Content-Type": "application/json"}

    data = {
        "yql": "select * from cnsl where rank({targetHits:6000}nearestNeighbor(text_embedding,tensor_tes), userQuery()) limit 30",
        "queryProfile": "MyProfile",
        "query": query,
        "timeout": "180s",
        "type": "weakAnd",
        "ranking": "hybrid",
        "ranking.softtimeout.enable": False,
        "input.query(tensor_tes)": tensor_tes,
        "presentation": {"bolding": True, "format": "json"},
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    workingdictionary = json.loads(response.text)

    hits_list = workingdictionary["root"]["children"]  # list of dicts, each hit a dict

    id_text_list = []  # list of dict
    # score_dict = {} not recording scores. need to retrieve from response object
    for hit in hits_list:
        hit_dict = {}
        id = hit["id"]
        # print(id)
        closest_index = int(
            list(
                hit["fields"]["matchfeatures"]["closest(text_embedding)"][
                    "cells"
                ].keys()
            )[0]
        )

        closest_para = hit["fields"]["text"][closest_index]

        hit_dict["id"] = id
        hit_dict["text"] = closest_para
        id_text_list.append(hit_dict)

    results = cohere_rr_response(id_text_list)

    desired_order_ids = [id_text_list[hit.index]["id"] for hit in results.results]
    order_mapping = {id_: index for index, id_ in enumerate(desired_order_ids)}
    reranked_hits_list = sorted(hits_list, key=lambda x: order_mapping[x["id"]])
    return reranked_hits_list


@app.post("/top_cases_progress")
def top_cases_progress(request: dict = Body(...)) -> str:
    reranked_hits_list = request.get("reranked_hits_list", [])
    no_best_hits = request.get("no_best_hits", 5)
    progress_string = "Cases being analysed:\n"
    for i in range(0, no_best_hits):
        case_name = reranked_hits_list[i]["fields"]["citation"]
        progress_string += case_name + "\n"
    progress_string += "..."
    return progress_string


@app.post("/llm_output")
def llm_output(request: dict = Body(...)) -> list[dict]:
    reranked_hits_list = request.get("reranked_hits_list", [])
    query = request.get("query", "")
    no_best_hits = request.get("no_best_hits", 5)

    def on_backoff(details):
        print(
            f"Retrying in {details['wait']:.1f} seconds, due to {details['exception']}"
        )

    @backoff.on_exception(backoff.expo, Exception, max_tries=30, on_backoff=on_backoff)
    async def gemini(text: str):
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=gemini_api_key)

        generation_config = {
            "temperature": 0.2,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 2048,
            "response_mime_type": "application/json",
        }

        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash-latest",
            generation_config=generation_config,
            safety_settings=safety_settings,
        )

        prompt_parts = [
            f"""
            {text}
            """,
        ]

        response = model.generate_content_async(prompt_parts)
        return await response

    async def process_case(i):
        case_name = reranked_hits_list[i]["fields"]["citation"]
        text_chunks = reranked_hits_list[i]["fields"]["text"]
        source = reranked_hits_list[i]["fields"]["source"]

        case_extract = "".join(text_chunks)

        prompt = f"""
        <your role>
        You are a legal analyst who is given a Question, and information about one legal Judgment. 
        The context information of that judgment comprises: the Citation (comprising the case name and the action number), and full text of the Judgment (i.e. Judgment text).
        Your job is to summarize and analyze the relevance or irrelevance of the Judgment to the Question.
        </your role>
        
        <context_information>
        The Question (which may or may not come with related additional context) is as follows:
        <question>
        {query}
        </question>
        
        The Citation is as follows:
        <citation>
        {case_name}
        </citation>

        The Judgment text is as follows:
        <judgment_text>
        {case_extract}
        </judgment_text>
        </context_information>
        
        <analysis>
        Set out your thinking and give your answer using this JSON schema:

        {{
            "type": "object",
            "properties": {{
                "citation": {{
                    "type": "string"
                }},
                "source": {{
                    "type": "string"
                }},
                "summary": {{
                    "type": "string"
                }},
                "thinking": {{
                    "type": "string"
                }},
                "relevance_analysis_draft": {{
                    "type": "string"
                }},
                "relevance_analysis": {{
                    "type": "string"
                }},
                "relevance_score": {{
                    "type": "integer"
                }},
            }}
        }}
            
        Fill in this JSON according the the following instructions:-
        
        "citation": {case_name}

        "source": {source}

        "summary": Give a short summary, one to three sentences long, of what the Judgment text discusses or decides.

        "thinking":
        Before proceeding to write your answers below, take a deep breath, and please think about it here, step-by-step. 
        Think carefully about the Question. Think carefully about what the Judgment text discusses and decides. Don't conflate or confuse the contents of the Question with the contents of the Judgment text. 
        After you are done thinking here, move on to write your extremely detailed and refined answers below. 

        "relevance_analysis_draft":
        Is the Judgment text directly, specifically, and highly relevant to the Question?
        If it is not directly, specifically, and highly relevant, explain why it is not in one sentence. Be very brief. 
        e.g. "The Judgment is not relevant, as the Judgment only decides or concerns ..."
        On the other hand, if the Judgment text is directly, specifically, and highly relevant to the Question, provide an extremely detailed and comprehensive analysis of how the Judgment text helps to answer the Question. 
        If the relevant part of the Judgment text cites other cases, statutes or textbooks which help to answer the Question, summarize the court's use and/or discussion of these authorities.
        e.g. "The Judgment is relevant, as the Judgment decides ... (at §1)). ... In support of this proposition, the case of Chow Xiao Ming v Great Profits Limited (HCA 1234/2000, 1 January 2001) is cited (§2) ..."

        "relevance_analysis":
        Read your "relevance_analysis_draft" carefully.
        Re-read the Judgment text carefully.
        Is there anything in your "relevance_analysis_draft" which should be corrected?
        Is there anything which needs to be added to the "relevance_analysis_draft"?
        Are there paragraph references (§x) which should be corrected or added?
        Write the refined, edited, and proof-read "relevance_analysis_draft" here.

        "relevance_score":
        On a scale from 0-100, being very precise, give an integer between 0-100 which scores the relevance of the Judgment text to the Question. Just give the integer, and nothing else. No further explanations.
        </analysis>
        
        <formatting>
        When writing your answers as the JSON values, ALWAYS follow these formatting and stylistic guidelines:
        - NEVER use markdown. Never use ** **.
        - When referring to the court which wrote the Case Extract, always refer to "the Court" or "the court". Do not refer to a specific court.
        - Make as many references to the paragraphs of the Judgment text as you can. Paragraph 1 should be referred to at the end of a sentence as (at §1). Paragraph 1-2 should be referred to at the end of a sentence as (§§1-2). You must adhere strictly to this style of paragraph referencing.
        </formatting>

        
        {{
        """

        class OutputSchema(BaseModel):
            citation: str
            source: str
            summary: str
            thinking: str
            relevance_analysis_draft: str
            relevance_analysis: str
            relevance_score: int

        max_retries = 5
        for i in range(0, max_retries + 1):
            try:
                response = await gemini(prompt)
                llm_output = response.text
                json_output = json.loads(llm_output)
                OutputSchema(**json_output)
                return json_output

            except Exception as e:
                print(e)
                print(f"going to retry: {i}")

        print("5 retries all failed")

    # number of cases to process
    async def rel_analysis():
        tasks = [process_case(i) for i in range(no_best_hits)]
        results = await asyncio.gather(*tasks)
        return results

    output_record = asyncio.run(rel_analysis())

    return output_record
