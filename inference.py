import os
import json
import re
import textwrap
from typing import List, Dict, Any

from openai import OpenAI
from pydantic import BaseModel

from hackathon.server.hackathon_environment import SupplyChainEnv
from hackathon.models import AgentAction

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
MAX_STEPS = 30
TEMPERATURE = 0.2
MAX_TOKENS = 1000

SYSTEM_PROMPT = textwrap.dedent("""
    You are a supply chain manager controlling 7 nodes and 3 products (21 total slots).
    You need to output order quantities and shipping methods for each slot to maximize fill rate and minimize costs.
    Output MUST be a valid JSON object matching this schema:
    {
      "order_quantities": [float, float, ...], // exactly 21 floats
      "shipping_methods": [int, int, ...]      // exactly 21 ints (0 for standard, 1 for express)
    }
    No explanations, no markdown blocks, just the JSON.
""").strip()


def build_user_prompt(step: int, observation: Any) -> str:
    state_vector = observation.state_vector
    fill_rate = observation.fill_rate
    prompt = textwrap.dedent(f"""
        Day: {step}
        Current Fill Rate: {fill_rate:.2f}
        State Vector (summarized context): {state_vector[:10]}... (truncated)
        Demand Forecast: {observation.demand_forecast}
        Inventory Levels: {observation.inventory_levels}
        
        Provide the next action as JSON. Keep orders reasonable based on the forecast.
    """).strip()
    return prompt


def parse_model_action(response_text: str) -> AgentAction:
    try:
        text = response_text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        data = json.loads(text.strip())
        
        order_qs = data.get("order_quantities", [0.0] * 21)
        ship_ms = data.get("shipping_methods", [0] * 21)
        
        if len(order_qs) < 21: order_qs += [0.0] * (21 - len(order_qs))
        if len(ship_ms) < 21: ship_ms += [0] * (21 - len(ship_ms))
        
        return AgentAction(order_quantities=order_qs[:21], shipping_methods=ship_ms[:21])
    except Exception as e:
        print(f"Failed to parse action: {e}. Using fallback.")
        return AgentAction(order_quantities=[0.0] * 21, shipping_methods=[0] * 21)


def grade_episode(state, task_name: str) -> float:
    fill_rate = state.fill_rate
    cost = state.total_cost
    
    score = fill_rate
    
    if task_name == "easy":
        if fill_rate > 0.7:
            score = 0.7 + (fill_rate - 0.7) * 0.3
        else:
            score = fill_rate * 0.5
    elif task_name == "medium":
        if fill_rate > 0.8:
            score = 0.8 + (fill_rate - 0.8) * 0.2
        else:
            score = fill_rate * 0.6
    elif task_name == "hard":
        carbon_penalty = min(0.2, state.carbon_footprint / 10000.0) 
        adjusted_score = fill_rate - carbon_penalty
        score = max(0.0, min(1.0, adjusted_score))
        
    return float(max(0.0, min(1.0, score)))


def run_task(client: OpenAI, task_name: str) -> float:
    print(f"\n--- Starting Task: {task_name} ---")
    env = SupplyChainEnv()
    
    try:
        obs = env.reset(difficulty=task_name, horizon=MAX_STEPS)
        print(f"Task initialized with difficulty: {task_name}")
        
        for step in range(1, MAX_STEPS + 1):
            user_prompt = build_user_prompt(step, obs)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    stream=False,
                )
                response_text = completion.choices[0].message.content or ""
            except Exception as exc:
                print(f"Model request failed ({exc}). Using fallback action.")
                response_text = ""
                
            action = parse_model_action(response_text)
            obs = env.step(action)
            
            if obs.done:
                print("Episode complete early.")
                break
                
        state = env.state
        score = grade_episode(state, task_name)
        print(f"Task '{task_name}' finished. Fill Rate: {state.fill_rate:.2f}, Cost: {state.total_cost:.2f}")
        print(f"Task Score (0.0-1.0): {score:.4f}")
        return score

    except Exception as e:
        print(f"Error during task {task_name}: {e}")
        return 0.0

def main():
    if not API_KEY:
        print("Warning: API_KEY/HF_TOKEN not set. Inference might fail.")
        
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "dummy")
    
    tasks = ["easy", "medium", "hard"]
    scores = {}
    
    for task in tasks:
        score = run_task(client, task)
        scores[task] = score
        
    print("\n===============================")
    print("Baseline Inference Scores:")
    for task, score in scores.items():
        print(f" - {task.capitalize()} Task: {score:.4f}")
    print("===============================")

if __name__ == "__main__":
    main()
