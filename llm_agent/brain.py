import json
import re
try:
    import ollama
except ImportError:
    ollama = None

class LLMBrain:
    """The high-level planner for the NeuroGenesis robot.
    
    Uses a local LLM to translate natural language commands into a strict
    JSON sequence of low-level skills.
    """

    def __init__(self, model_name="llama3"):
        self.model_name = model_name
        if ollama is None:
            raise ImportError("The 'ollama' Python package is not installed. Please install it with `pip install ollama`.")

    def parse_command(self, command: str, available_objects: list) -> list:
        """Parse a natural language command into a list of skill dictionaries."""
        
        system_prompt = f"""You are the brain of a household mobile manipulator robot.
Your job is to translate the user's natural language command into a JSON array of actions.

Available skills:
- "navigate_to" (moves the robot to an object)
- "pick" (picks up an object)
- "place" (places the held object at a location)
- "toggle" (turns a smart device on or off, like a TV or lamp)
- "clean" (cleans/sweeps dirt, dust, or spills off the floor)

Available objects in the room:
{json.dumps(available_objects)}

Rules:
1. You must navigate to an object before you can pick it up, toggle it, or clean it.
2. You must pick an object before you can place it.
3. You must navigate to the target location before you can place the object there.
4. Use "toggle" for devices like "tv" and "lamp". Do NOT use "pick" on them.
5. Use "clean" for "dirt" or spills. Do NOT use "pick" on dirt.
6. Output strictly valid JSON as an array of objects with "skill" and "target" keys.
7. Do not include any explanations, greetings, or markdown code blocks around the JSON. Just the raw JSON array.

Example Input: "Put the cereal box on the dining table"
Example Output:
[
  {{"skill": "navigate_to", "target": "cereal_box"}},
  {{"skill": "pick", "target": "cereal_box"}},
  {{"skill": "navigate_to", "target": "dining_table"}},
  {{"skill": "place", "target": "dining_table"}}
]

Example Input: "Turn on the TV"
Example Output:
[
  {{"skill": "navigate_to", "target": "tv"}},
  {{"skill": "toggle", "target": "tv"}}
]

Example Input: "Clean the dirt on the floor"
Example Output:
[
  {{"skill": "navigate_to", "target": "dirt"}},
  {{"skill": "clean", "target": "dirt"}}
]
"""

        print(f"[*] Sending command to {self.model_name}...")
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": command}
                ]
            )
        except Exception as e:
            print(f"[!] Error communicating with Ollama: {e}")
            print("Make sure you have installed Ollama and run `ollama run llama3` in your terminal.")
            return []

        raw_content = response['message']['content'].strip()
        
        # Cleanup: sometimes LLMs still wrap in ```json ... ``` despite instructions
        if raw_content.startswith("```json"):
            raw_content = raw_content[7:]
        if raw_content.startswith("```"):
            raw_content = raw_content[3:]
        if raw_content.endswith("```"):
            raw_content = raw_content[:-3]
        
        raw_content = raw_content.strip()

        try:
            plan = json.loads(raw_content)
            print(f"[*] LLM generated plan: {json.dumps(plan, indent=2)}")
            return plan
        except json.JSONDecodeError:
            print(f"[!] Failed to parse LLM output as JSON. Raw output:\n{raw_content}")
            # Try to find JSON array with regex
            match = re.search(r'\[.*\]', raw_content, re.DOTALL)
            if match:
                try:
                    plan = json.loads(match.group(0))
                    print(f"[*] Recovered plan via regex: {json.dumps(plan, indent=2)}")
                    return plan
                except:
                    pass
            return []
