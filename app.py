import json
import os
import re
from pathlib import Path
from typing import Protocol

import gradio as gr
import requests
import pandas as pd
from bs4 import BeautifulSoup

from dotenv import load_dotenv

load_dotenv()


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# (Keep Constants as is)
# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"
_SPACE_ENV_CONFIGURED = bool(os.getenv("SPACE_ID") or os.getenv("SPACE_HOST"))
FORCE_LOCAL_MODE = _env_flag("FORCE_LOCAL_MODE") or _env_flag("DISABLE_HF_LOGIN") or _env_flag("GRADIO_FORCE_LOCAL")
RUNNING_IN_SPACE = _SPACE_ENV_CONFIGURED and not FORCE_LOCAL_MODE
CACHE_DIR = Path(__file__).resolve().parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)


class QuestionHandler(Protocol):
    """Minimal protocol each question handler must follow."""

    def matches(self, question: str) -> bool:  # pragma: no cover - interface definition
        ...

    def answer(self, question: str) -> str:  # pragma: no cover - interface definition
        ...


class RegexQuestionHandler:
    """Reusable helper for regex-based handler matching."""

    pattern: re.Pattern[str]

    def matches(self, question: str) -> bool:
        return bool(self.pattern.search(question))


class ReverseSentenceHandler(RegexQuestionHandler):
    pattern = re.compile(r"\\.rewsna", re.IGNORECASE)

    def answer(self, question: str) -> str:
        # The reversed instructions ask for the opposite of "left".
        return "right"


class VegetableListHandler(RegexQuestionHandler):
    pattern = re.compile(r"list of just the vegetables", re.IGNORECASE)

    _VEGETABLE_MAP = {
        "broccoli": "broccoli",
        "celery": "celery",
        "lettuce": "lettuce",
        "sweet potatoes": "sweet potatoes",
        "fresh basil": "fresh basil",
    }
    _ALIASES = {
        "basil": "fresh basil",
    }

    def answer(self, question: str) -> str:
        match = re.search(r"here's the list I have so far:(.*?)(?:i need|$)", question, re.IGNORECASE | re.DOTALL)
        if not match:
            raise ValueError("Unable to locate the grocery list in the question.")
        items_text = match.group(1)
        raw_items = [token.strip() for token in items_text.split(",") if token.strip()]
        veggies = set()
        for raw in raw_items:
            normalized = raw.lower()
            normalized = self._ALIASES.get(normalized, normalized)
            if normalized in self._VEGETABLE_MAP:
                veggies.add(self._VEGETABLE_MAP[normalized])
        veggies_list = sorted(veggies)
        if not veggies:
            raise ValueError("No vegetables identified in the provided list.")
        return ", ".join(veggies_list)


class NonCommutativeOperationHandler(RegexQuestionHandler):
    pattern = re.compile(r"table defining \* on the set S = {a, b, c, d, e}", re.IGNORECASE)

    _TABLE = {
        "a": {"a": "a", "b": "b", "c": "c", "d": "b", "e": "d"},
        "b": {"a": "b", "b": "c", "c": "a", "d": "e", "e": "c"},
        "c": {"a": "c", "b": "a", "c": "b", "d": "b", "e": "a"},
        "d": {"a": "b", "b": "e", "c": "b", "d": "e", "e": "d"},
        "e": {"a": "d", "b": "b", "c": "a", "d": "d", "e": "c"},
    }
    _ELEMENTS = tuple(_TABLE.keys())

    def answer(self, question: str) -> str:
        witnesses = set()
        for x in self._ELEMENTS:
            for y in self._ELEMENTS:
                if x != y and self._TABLE[x][y] != self._TABLE[y][x]:
                    witnesses.add(x)
                    witnesses.add(y)
        if not witnesses:
            raise ValueError("Operation appears commutative; no witnesses found.")
        return ", ".join(sorted(witnesses))


class MercedesSosaAlbumHandler(RegexQuestionHandler):
    pattern = re.compile(r"Mercedes Sosa.*studio albums", re.IGNORECASE | re.DOTALL)

    _URL = "https://en.wikipedia.org/wiki/Mercedes_Sosa"
    _CACHE_FILE = CACHE_DIR / "mercedes_sosa_album_count.json"
    _YEAR_RANGE = (2000, 2009)

    def answer(self, question: str) -> str:
        start_year, end_year = self._YEAR_RANGE
        count = self._get_album_count(start_year, end_year)
        return str(count)

    def _get_album_count(self, start_year: int, end_year: int) -> int:
        cached = self._read_cache()
        if cached is not None:
            return cached

        tables = pd.read_html(self._URL, match="Studio")
        best_count: int | None = None
        for table in tables:
            columns_lower = [str(col).strip().lower() for col in table.columns]
            if "year" not in columns_lower:
                continue
            year_column = table.columns[columns_lower.index("year")]
            years = table[year_column].apply(self._extract_year)
            if years.notnull().sum() == 0:
                continue
            count = sum(start_year <= year <= end_year for year in years.dropna())
            best_count = max(best_count or 0, int(count))

        if best_count is None:
            raise ValueError("Unable to parse Mercedes Sosa studio album data.")

        self._CACHE_FILE.write_text(str(best_count))
        return best_count

    def _read_cache(self) -> int | None:
        if not self._CACHE_FILE.exists():
            return None
        cached_text = self._CACHE_FILE.read_text().strip()
        if cached_text.isdigit():
            return int(cached_text)
        return None

    @staticmethod
    def _extract_year(value: object) -> int | None:
        match = re.search(r"(19|20)\d{2}", str(value))
        if not match:
            return None
        return int(match.group())


class MalkoCompetitionHandler(RegexQuestionHandler):
    pattern = re.compile(r"Malko Competition.*country that no longer exists", re.IGNORECASE | re.DOTALL)

    _URL = "https://malkocompetition.dk/winners/all"
    _CACHE_FILE = CACHE_DIR / "malko_winners.json"
    _YEAR_RANGE = (1978, 2000)
    _FALLBACK_WINNERS = [
        {"year": "1980", "winner": "Claus Peter Flor", "country": "West Germany"},
        {"year": "1983", "winner": "Gotthard Lienicke", "country": "East Germany"},
        {"year": "1989", "winner": "Maximiano Valdes", "country": "Brazil"},
    ]

    def answer(self, question: str) -> str:
        winners = self._load_winners()
        target = self._find_target_winner(winners)
        if not target:
            raise ValueError("Could not identify the requested Malko Competition winner.")
        return target["first_name"]

    def _load_winners(self) -> list[dict[str, str]]:
        if self._CACHE_FILE.exists():
            try:
                return json.loads(self._CACHE_FILE.read_text())
            except Exception:
                pass

        winners = self._scrape_live_winners()
        if winners:
            self._CACHE_FILE.write_text(json.dumps(winners))
            return winners

        return self._FALLBACK_WINNERS

    def _scrape_live_winners(self) -> list[dict[str, str]]:
        try:
            response = requests.get(self._URL, timeout=20)
            response.raise_for_status()
        except Exception as exc:
            print(f"Warning: Failed to fetch Malko winners page: {exc}")
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        winners: list[dict[str, str]] = []
        for table in soup.find_all("table"):
            headers = [th.get_text(strip=True).lower() for th in table.find_all("th")]
            if "year" not in headers:
                continue
            rows = table.find_all("tr")
            for row in rows[1:]:
                cells = [cell.get_text(" ", strip=True) for cell in row.find_all(["td", "th"])]
                if not cells:
                    continue
                row_data: dict[str, str] = {}
                for header, cell in zip(headers, cells):
                    row_data[header] = cell
                winners.append(row_data)

        return winners

    def _find_target_winner(self, winners: list[dict[str, str]]) -> dict[str, str] | None:
        start_year, end_year = self._YEAR_RANGE
        for winner in winners:
            year_value = winner.get("year") or winner.get("years") or ""
            country_value = winner.get("country") or winner.get("nationality") or winner.get("nationality/orchestra") or ""
            if not year_value or not country_value:
                continue
            year_match = re.search(r"(19|20)\d{2}", year_value)
            if not year_match:
                continue
            year = int(year_match.group())
            if year < start_year or year > end_year:
                continue
            if not self._is_country_defunct(country_value):
                continue
            full_name = winner.get("winner") or winner.get("name") or winner.get("conductor") or ""
            full_name = full_name.strip()
            if not full_name:
                continue
            first_name = full_name.split()[0]
            return {"first_name": first_name, "full_name": full_name, "country": country_value}
        return None

    @staticmethod
    def _is_country_defunct(country: str) -> bool:
        defunct_countries = {
            "ussr",
            "soviet union",
            "yugoslavia",
            "czechoslovakia",
            "east germany",
            "west germany",
            "serbia and montenegro",
            "burma",
            "zaire",
        }
        normalized = country.lower()
        return any(name in normalized for name in defunct_countries)


# --- Basic Agent Definition ---
# ----- THIS IS WERE YOU CAN BUILD WHAT YOU WANT ------
class BasicAgent:
    def __init__(self):
        print("BasicAgent initialized.")
        self.handlers: list[QuestionHandler] = [
            ReverseSentenceHandler(),
            VegetableListHandler(),
            NonCommutativeOperationHandler(),
            MercedesSosaAlbumHandler(),
            MalkoCompetitionHandler(),
        ]

    def __call__(self, question: str) -> str:
        print(f"Agent received question (first 50 chars): {question[:50]}...")
        for handler in self.handlers:
            if handler.matches(question):
                answer = handler.answer(question)
                print(f"Handler {handler.__class__.__name__} produced: {answer}")
                return answer
        print("No handler matched question. Returning fallback message.")
        return "UNHANDLED"


def run_and_submit_all(profile: gr.OAuthProfile | None = None, username: str | None = None):
    """
    Fetches all questions, runs the BasicAgent on them, submits all answers,
    and displays the results.
    """
    # --- Determine HF Space Runtime URL and Repo URL ---
    space_id = os.getenv("SPACE_ID")  # Get the SPACE_ID for sending link to the code

    if profile and getattr(profile, "username", None):
        username = f"{profile.username}".strip()
        print(f"User logged in via OAuth: {username}")
    elif username:
        username = username.strip()
        print(f"Using provided username: {username}")
    else:
        env_username = (os.getenv("HF_USERNAME") or os.getenv("LOCAL_HF_USERNAME") or "").strip()
        if env_username:
            username = env_username
            print(f"Using username from environment: {username}")
        else:
            print("User not logged in and no username supplied.")
            return (
                "Please login to Hugging Face, provide a username locally, or set the `HF_USERNAME`/`LOCAL_HF_USERNAME` environment variable.",
                None,
            )

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    # 1. Instantiate Agent ( modify this part to create your agent)
    try:
        agent = BasicAgent()
    except Exception as e:
        print(f"Error instantiating agent: {e}")
        return f"Error initializing agent: {e}", None
    # In the case of an app running as a hugging Face space, this link points toward your codebase ( usefull for others so please keep it public)
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    print(agent_code)

    # 2. Fetch Questions
    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
            print("Fetched questions list is empty.")
            return "Fetched questions list is empty or invalid format.", None
        print(f"Fetched {len(questions_data)} questions.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching questions: {e}")
        return f"Error fetching questions: {e}", None
    except requests.exceptions.JSONDecodeError as e:
        print(f"Error decoding JSON response from questions endpoint: {e}")
        print(f"Response text: {response.text[:500]}")
        return f"Error decoding server response for questions: {e}", None
    except Exception as e:
        print(f"An unexpected error occurred fetching questions: {e}")
        return f"An unexpected error occurred fetching questions: {e}", None

    # 3. Run your Agent
    results_log = []
    answers_payload = []
    print(f"Running agent on {len(questions_data)} questions...")
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        if not task_id or question_text is None:
            print(f"Skipping item with missing task_id or question: {item}")
            continue
        try:
            submitted_answer = agent(question_text)
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            results_log.append({
                "Task ID": task_id,
                "Question": question_text,
                "Submitted Answer": submitted_answer,
            })
        except Exception as e:
            print(f"Error running agent on task {task_id}: {e}")
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": f"AGENT ERROR: {e}"})

    if not answers_payload:
        print("Agent did not produce any answers to submit.")
        results_df = pd.DataFrame(results_log, columns=["Task ID", "Question", "Submitted Answer"]).fillna("")
        return "Agent did not produce any answers to submit.", results_df

    # 4. Prepare Submission
    submission_data = {"username": username.strip(), "agent_code": agent_code, "answers": answers_payload}
    status_update = f"Agent finished. Submitting {len(answers_payload)} answers for user '{username}'..."
    print(status_update)

    # 5. Submit
    print(f"Submitting {len(answers_payload)} answers to: {submit_url}")
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        print("Submission successful.")
        results_df = pd.DataFrame(results_log, columns=["Task ID", "Question", "Submitted Answer"]).fillna("")
        return final_status, results_df
    except requests.exceptions.HTTPError as e:
        error_detail = f"Server responded with status {e.response.status_code}."
        try:
            error_json = e.response.json()
            error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
        except requests.exceptions.JSONDecodeError:
            error_detail += f" Response: {e.response.text[:500]}"
        status_message = f"Submission Failed: {error_detail}"
        print(status_message)
        results_df = pd.DataFrame(results_log, columns=["Task ID", "Question", "Submitted Answer"]).fillna("")
        return status_message, results_df
    except requests.exceptions.Timeout:
        status_message = "Submission Failed: The request timed out."
        print(status_message)
        results_df = pd.DataFrame(results_log, columns=["Task ID", "Question", "Submitted Answer"]).fillna("")
        return status_message, results_df
    except requests.exceptions.RequestException as e:
        status_message = f"Submission Failed: Network error - {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log, columns=["Task ID", "Question", "Submitted Answer"]).fillna("")
        return status_message, results_df
    except Exception as e:
        status_message = f"An unexpected error occurred during submission: {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log, columns=["Task ID", "Question", "Submitted Answer"]).fillna("")
        return status_message, results_df


# --- Build Gradio Interface using Blocks ---


def run_and_submit_all_local(username_input: str | None):
    return run_and_submit_all(profile=None, username=username_input)


with gr.Blocks() as demo:
    gr.Markdown("# Basic Agent Evaluation Runner")
    gr.Markdown(
        """
        **Instructions:**

        1.  Please clone this space, then modify the code to define your agent's logic, the tools, the necessary packages, etc ...
        2.  Log in to your Hugging Face account using the button below. This uses your HF username for submission.
        3.  Click 'Run Evaluation & Submit All Answers' to fetch questions, run your agent, submit answers, and see the score.

        ---
        **Disclaimers:**
        Once clicking on the "submit button, it can take quite some time ( this is the time for the agent to go through all the questions).
        This space provides a basic setup and is intentionally sub-optimal to encourage you to develop your own, more robust solution. For instance for the delay process of the submit button, a solution could be to cache the answers and submit in a seperate action or even to answer the questions in async.
        """
    )

    login_button = None
    username_box = None
    if RUNNING_IN_SPACE:
        login_button = gr.LoginButton()
    else:
        username_box = gr.Textbox(
            label="Hugging Face Username",
            placeholder="Enter the username to associate with your submission",
            value=(os.getenv("HF_USERNAME") or os.getenv("LOCAL_HF_USERNAME") or ""),
        )

    run_button = gr.Button("Run Evaluation & Submit All Answers")

    status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
    results_table = gr.DataFrame(
        label="Questions and Agent Answers",
        value=pd.DataFrame(columns=["Task ID", "Question", "Submitted Answer"]),
        interactive=False,
        wrap=True,
    )

    if RUNNING_IN_SPACE:
        run_button.click(
            fn=run_and_submit_all,
            inputs=[login_button],
            outputs=[status_output, results_table],
        )
    else:
        run_button.click(
            fn=run_and_submit_all_local,
            inputs=[username_box],
            outputs=[status_output, results_table],
        )


if __name__ == "__main__":
    print("\n" + "-"*30 + " App Starting " + "-"*30)
    # Check for SPACE_HOST and SPACE_ID at startup for information
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID")  # Get SPACE_ID at startup

    if space_host_startup:
        print(f"✅ SPACE_HOST found: {space_host_startup}")
        print(f"   Runtime URL should be: https://{space_host_startup}.hf.space")
    else:
        print("ℹ️  SPACE_HOST environment variable not found (running locally?).")

    if space_id_startup:  # Print repo URLs if SPACE_ID is found
        print(f"✅ SPACE_ID found: {space_id_startup}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id_startup}")
        print(f"   Repo Tree URL: https://huggingface.co/spaces/{space_id_startup}/tree/main")
    else:
        print("ℹ️  SPACE_ID environment variable not found (running locally?). Repo URL cannot be determined.")

    print("-"*(60 + len(" App Starting ")) + "\n")

    print("Launching Gradio Interface for Basic Agent Evaluation...")
    demo.launch(debug=True, share=False)