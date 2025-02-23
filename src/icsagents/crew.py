# crew.py
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
# from dotenv import load_dotenv

# load_dotenv()

@CrewBase
class Icsagents:
    """Icsagents crew for mathematical problem solving"""

    def __init__(self, context_data=None):
        self.agents_config = 'config/agents.yaml'
        self.tasks_config = 'config/tasks.yaml'
        self.context_data = context_data or {}
        self.ollama_llm = LLM(
            model='ollama/tinyllama',
            base_url='http://ollama:11434/api/generate',
        )

    @agent
    def math_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['math_researcher'],
            llm=self.ollama_llm,
            verbose=True,
        )

    @agent
    def math_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['math_analyst'],
            llm=self.ollama_llm,
            verbose=True,
        )

    @task
    def research_task(self, topic: str) -> Task:
        return Task(
            config=self.tasks_config['problem_breakdown_task'],
            description=f"Research and break down the topic: {topic}",
            context={"topic": topic, **self.context_data}  # Context passed to task
        )

    @task
    def reporting_task(self, topic: str) -> Task:
        return Task(
            config=self.tasks_config['solution_generation_task'],
            description=f"Generate solution report for: {topic}",
            context={"topic": topic, **self.context_data}  # Context passed to task
        )

    @crew
    def crew(self, topic: str) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=[
                self.research_task(topic),
                self.reporting_task(topic)
            ],
            process=Process.sequential,
            verbose=True
        )