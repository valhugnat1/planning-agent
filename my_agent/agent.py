from my_agent.utils.nodes import plan_step, execute_step, replan_step, routing_step, clean_response_step
from my_agent.utils.state import PlanExecute
from langgraph.graph import StateGraph, START, END
from my_agent.utils.naming import PLANNER, AGENT, REPLAN, ROUTING, CLEAN_RESPONSE


workflow = StateGraph(PlanExecute)

# Add the plan node
workflow.add_node(PLANNER, plan_step)
workflow.add_node(AGENT, execute_step)
workflow.add_node(REPLAN, replan_step)
workflow.add_node(ROUTING, routing_step)
workflow.add_node(CLEAN_RESPONSE, clean_response_step)

workflow.add_edge(START, PLANNER)

# From plan we go to agent
workflow.add_edge(PLANNER, AGENT)

# From agent, we replan
workflow.add_edge(AGENT, ROUTING)


def should_end(state: PlanExecute):
    if state["routed_to"] == REPLAN:
        return REPLAN
    else:
        return CLEAN_RESPONSE

workflow.add_conditional_edges(
    ROUTING,
    # Next, we pass in the function that will determine which node is called next.
    should_end,
    [REPLAN, CLEAN_RESPONSE],
)

workflow.add_edge(REPLAN, AGENT)
workflow.add_edge(CLEAN_RESPONSE, END)

graph = workflow.compile()
graph.name = "Agent"