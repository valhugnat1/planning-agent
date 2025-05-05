from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
# from langchain_mistralai.chat_models import ChatMistralAI

from my_agent.utils.prompts import  planner_prompt, replanner_prompt, prompt_executor, routing_prompt, Plan, Act, clean_response_prompt
from my_agent.utils.state import PlanExecute
from my_agent.utils.tools import tools

from dotenv import load_dotenv
import os


load_dotenv()

llm = ChatOpenAI(
    base_url=os.getenv("SCW_GENERATIVE_APIs_ENDPOINT"),
    api_key=os.getenv("SCW_SECRET_KEY"), 
    # model="mistral-small-3.1-24b-instruct-2503",
    model="llama-3.3-70b-instruct",
    temperature=0
)

# llm_code = ChatOpenAI(
#     base_url=os.getenv("SCW_GENERATIVE_APIs_ENDPOINT"),
#     api_key=os.getenv("SCW_SECRET_KEY"), 
#     model="qwen2.5-coder-32b-instruct",
#     temperature=0.5
# )

# llm = ChatAnthropic(
#     api_key=os.getenv("ANTHROPIC_API_KEY"),
#     model="claude-3-5-sonnet-20241022"
# )

# llm = ChatMistralAI(
#     api_key=os.getenv("MISTRAL_API_KEY"),
#     model="mistral-large-latest"
# )


# llm = llm = ChatOpenAI(
#     api_key=os.getenv("FIREWORKS_API_KEY"),
#     base_url="https://api.fireworks.ai/inference/v1",
#     model="accounts/fireworks/models/qwen3-30b-a3b",
# )


llm = ChatOpenAI(
    api_key="fake",
    base_url="http://localhost:8000",
    model="accounts/fireworks/models/qwen3-30b-a3b",
    streaming=False
)


agent_executor = create_react_agent(llm, tools, prompt=prompt_executor)

planner = planner_prompt | llm.with_structured_output(Plan)
replanner = replanner_prompt | llm.with_structured_output(Plan)
route = routing_prompt | llm.with_structured_output(Act)
clean_response = clean_response_prompt | llm


async def execute_step(state: PlanExecute):
    plan = state["plan"]
    plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
    task = plan[0]
    task_formatted = f"""For the following plan:
{plan_str}\n\nYou are tasked with executing step {1}, {task}.
Don't ask premision to execute the task. Just execute it."""
    agent_response = await agent_executor.ainvoke(
        {"messages": [("user", task_formatted)]}
    )
    return {
        "past_steps": [(task, agent_response["messages"][-1].content)],
    }


async def plan_step(state: PlanExecute):
    plan = await planner.ainvoke({"messages": [("user", state["input"])]})

    return {"plan": plan.steps}



async def replan_step(state: PlanExecute):
    replan = await replanner.ainvoke(state)

    return {"plan": replan.steps}


async def routing_step(state: PlanExecute):
    output = await route.ainvoke(state)
    print (output)
    return {"routed_to": output.route}



async def clean_response_step(state: PlanExecute):
    output = await clean_response.ainvoke(state)
    print (output)
    return {"response": output}
