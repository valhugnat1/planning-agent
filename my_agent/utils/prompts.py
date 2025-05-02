from langchain_core.prompts import ChatPromptTemplate
from typing import List, Literal, TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import END
from my_agent.utils.naming import PLANNER, AGENT, REPLAN, ROUTING, CLEAN_RESPONSE



prompt_executor = "You are a helpful assistant."

planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps. \
Do not return the answer of any step in the plan. \
Use JSON format. \
Example:  \
objective: Finding the director of a specific film and then exploring their filmography further.
  "steps": [ \
    "Find the director of the movie 'Pulp Fiction'. The director's name will be referred to as DirectorName", \
    "Search for another famous movie directed by DirectorName" \
  ] \
""",
        ),
        ("placeholder", "{messages}"),
    ]
)

replanner_prompt = ChatPromptTemplate.from_template(
    """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

Your objective was this:
{input}

Your original plan was this:
{plan}

You have currently done the follow steps:
{past_steps}

Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan.
Respond with JSON format if it's a plan, respopnse in plain TEXT if it's final anwser"""
)

routing_prompt = ChatPromptTemplate.from_template(
    """Choose """+CLEAN_RESPONSE+""" if you have all the information to make the objective or """+REPLAN+""" if you need to more information to make the objective. 
Your objective was this:
{input}

You have currently done the follow information:
{past_steps}
"""
)
clean_response_prompt = ChatPromptTemplate.from_template(
"""Clean the response to make it more readable and friendly.
Your objective was this:
{input}

Your original plan was this:
{plan}
You have currently done the follow steps:
{past_steps}
"""
)

class Plan(BaseModel):
    """Plan to follow in future"""

    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )

class Response(BaseModel):
    """Response to user."""

    response: str
    
class Act(BaseModel):
    """Action to perform."""

    # action: Response = Field(
    #     description="Action to perform. If you want to respond to user, use Response. "
    #     "If you need to further use tools to get the answer, use Plan."
    # )
    route: Literal[REPLAN, CLEAN_RESPONSE] = Field(
        description="Use JSON format to respond."
        f"Whether to replan or not. If you want to replan, use {REPLAN}. "
        f"If you want to finish and respond to user, use {CLEAN_RESPONSE}."
        "Use JSON format to respond."
    )

