# Gemini Code Assistant Instructions for IS-GOAT-BURNING

## 1. Role and Directives

You are an AI assistant maintaining the "Is the Gävle Goat Burning?" project. Your purpose is to generate code that is correct, performant, and strictly adheres to the project's architecture and rules. This document is your primary set of instructions.

## 2. Initial Context Gathering

Before addressing any specific task, you **MUST** first familiarize yourself with the project's high-level context. This information is located in the `README.md` file.

*   **Action:** Use the `ReadFile` tool to read the project's `README.md`.
*   **Purpose:** The `README.md` file describes the project's purpose, key features, and user-facing setup. This `GEMINI.md` file contains the developer-facing rules. You need both to work effectively.

## 3. GitHub Issue Workflow

When you are asked to resolve a GitHub issue, you **MUST** follow this structured, three-phase process.

### Phase 1: Understand and Plan

1.  **Request Context:** Begin by asking the user to provide the full context of the GitHub issue, including its title, body, and any relevant comments.
2.  **Formulate a Plan:** After reviewing the issue, formulate a clear, step-by-step plan of action. Your plan **MUST** include:
    *   A summary of the goal (e.g., "My goal is to add a new Slack notification service as described in issue #42.").
    *   A list of files you anticipate creating or modifying.
    *   The sequence of actions you will take.
3.  **Confirm the Plan:** Present the plan to the user and ask for their approval before you begin writing any code.

### Phase 2: Implement and Validate

1.  **Execute the Plan:** Follow your approved plan, using the `ReadFile`, `Edit`, and `Shell` tools as required.
2.  **Adhere to All Rules:** During implementation, you **MUST** adhere to all rules specified in the "Core Principles", "Global Rules", "Code Style", and "Directory- and File-Specific Instructions" sections of this document.
3.  **Validate Changes:** After making your code changes, you are **REQUIRED** to run the `pre-commit` validation as specified in the "Code Style and Quality" section.

### Phase 3: Summarize and Propose Commit

1.  **Provide a Summary:** Once the implementation is complete and validated, provide a clear summary of the work you have done.
2.  **Propose a Commit Message:** You **MUST** generate a well-formatted commit message that follows the "Conventional Commits" standard.
    *   **Format:** `<type>(<scope>): <description>`
    *   **Types:** `feat` (new feature), `fix` (bug fix), `docs` (documentation), `style`, `refactor`, `test`, `chore`.
    *   **Scope:** The module affected (e.g., `notifications`, `docker`, `fire-detection`).
    *   **Example Commit Message:**
        ```
        feat(notifications): add Slack notification service

        Implement a new SendToSlack class in on_fire_actions/ to send
        notifications to a configured Slack webhook when a fire is detected.
        Configuration is loaded from the .env file via setting.py.

        Closes #42
        ```
3.  **Propose a Pull Request Description:** If applicable, also provide a template for the Pull Request body.
    *   **Example PR Description:**
        ```
        This PR resolves issue #42 by adding a new notification service for Slack.

        **Changes:**
        - Created `on_fire_actions/send_to_slack.py`.
        - Updated `burning_goat_detection.py` to initialize and use the new service.
        - Added `SLACK_HOOK` to `.env.example` and `setting.py`.
        ```

## 4. Responding to Pull Request Reviews

When asked to apply changes from a pull request review, you **MUST** act as a developer by following this three-phase process.

### Phase 1: Ingest and Plan

1.  **Request Context:** You **MUST** ask the user to provide the following:
    *   The complete `git diff` of the code that was reviewed.
    *   A list of all review comments, grouped by filename.
2.  **Formulate a Plan:** You **MUST** process each review comment as a distinct sub-task. Create a checklist of the changes you will make.
    *   **Example Plan:**
        ```
        My plan is to address the review comments as follows:
        - [ ] In `on_fire_actions/send_email.py`: Change the `timeout` from 1 to 3 seconds.
        - [ ] In `fire_detection/utils.py`: Add a comment explaining the HSV color range.
        - [ ] In `README.md`: Fix the typo in the Docker run command.
        ```
3.  **Confirm the Plan:** Present this checklist to the user for approval before you begin editing files.

### Phase 2: Execute and Validate

1.  **Iterate and Implement:** Address each item on your plan one by one, using the `ReadFile` and `Edit` tools for each file modification.
2.  **Final Validation:** After all changes have been applied, you are **REQUIRED** to run a final validation using the `Shell` tool:
    ```bash
    pre-commit run --all-files
    ```
3.  **Debugging Loop:** If validation fails, analyze the error, state a hypothesis, propose a fix, and re-run validation until it passes.

### Phase 3: Summarize and Report

1.  **Confirm Completion:** Present your initial checklist again, this time with all items checked off, to confirm you have addressed all review feedback.
2.  **Propose Commit Message:** Generate a single, well-formatted commit message that summarizes the changes.
    *   **Example Commit Message:**
        ```
        refactor: apply suggestions from PR review

        - Increased email timeout for better reliability.
        - Improved code comments in the fire detection utility.
        - Corrected a typo in the documentation.
        ```
3.  **Generate Commit Message:** You **MUST** generate a well-formatted commit message that follows the "Conventional Commits" standard.
4.  **Propose PR Comment:** Provide a template for the user to post back on the GitHub PR.
    *   **Example PR Comment:**
        ```
        Thank you for the review! I have addressed all your feedback and pushed the changes. Please take another look when you have a moment.
        ```

## 5. Security Mandates

*   **NEVER log secrets:** Under no circumstances should credentials, API keys, or personal information be logged. When logging objects or data, ensure sensitive fields are excluded.
*   **Sanitize Shell Inputs:** Be extremely cautious with commands passed to the `Shell` tool. Never pass user-provided or dynamically generated strings directly to shell commands without validation.
*   **Validate Webhook URLs:** Before using a Discord or other webhook URL, ensure it has a valid format.

## 6. Testing Guidelines

*   **Location:** All tests should be placed in a `tests/` directory, mirroring the main project structure (e.g., `tests/on_fire_actions/test_send_email.py`).
*   **Framework:** Use `pytest` for running tests and `unittest.mock` for mocking external services like SMTP servers or Discord webhooks.
*   **Requirement:** Any new feature (e.g., a new notification service) or significant bug fix **SHOULD** be accompanied by corresponding tests that validate its behavior.
*   **Execution:** Instruct the user on how to run tests using `pytest`.

## 7. Core Principles

These are the fundamental architectural philosophies of this project. Adhere to them in all your work.

*   **Asynchronous First:** The entire application is built on `asyncio`. All new I/O operations you introduce (networking, file access) **MUST** be non-blocking and integrate into the existing event loop.
*   **Configuration via Environment:** All settings are managed through environment variables. **DO NOT** hardcode configuration values. `setting.py` is the single source of truth for accessing these values in the application.
*   **Strict Modularity:** Major components are isolated in their respective directories (`fire_detection`, `on_fire_actions`, `stream_recording`). You **MUST** respect this separation of concerns.

## 8. Global Rules

These rules are non-negotiable and apply to the entire project.

1.  **Configuration Workflow:** When adding a new configuration option:
    1.  Add the variable with a default value to `.env.example`.
    2.  Load it in `setting.py` using the appropriate `env()` method (e.g., `env.bool()`, `env.list()`).
    3.  Import the setting from `setting.py` where needed. **DO NOT** call `os.environ` or `env()` anywhere else.

2.  **Dependency Management:** To add a new Python dependency:
    1.  Add the package to `requirements.txt`.
    2.  If the package is not specific to the CUDA environment, you **MUST** also add it to `requirements_cuda.txt`.
    3.  Add it to the `[project]` section in `pyproject.toml`.
    4.  If the package is for development purposes, add it to `requirements_dev.txt` and to the `[project.optional-dependencies]` section in `pyproject.toml`.

3.  **Idempotent Actions:** All fire-response actions **MUST** be wrapped by the `OnceAction` class in `burning_goat_detection.py` to ensure they are triggered only once per execution.
4.  **Error Handling and Logging:**
    *   Use the standard `logging` module configured in the project.
    *   `logger.info()`: For routine lifecycle events (e.g., "Starting new video chunk").
    *   `logger.warning()`: For recoverable errors or unexpected situations that do not stop the application (e.g., "Frame queue is full, dropping frame").
    *   `logger.error()`: For critical, unrecoverable errors that may terminate a process (e.g., "Authentication failed").
    *   Use `try...except` blocks to gracefully handle potential exceptions (e.g., `TimeoutError`, `SMTPAuthenticationError`).


## 9. Code Style and Quality

This project enforces a strict code style using `ruff` and `pre-commit`.

*   **MANDATORY:** Before finalizing any code changes, you **MUST** validate them by executing the following command from the project root using the `Shell` tool:
    ```bash
    pre-commit run --all-files
    ```
*   If the user has not set up the environment, you MUST instruct them to run `pip install -r requirements_dev.txt` and `pre-commit install` first.
*   **Style Adherence:** Your code must follow PEP 8, use double quotes for strings, and conform to the `isort` configuration in `pyproject.toml`.

## 10. Tool Usage Directives

*   **`ReadFile`:** **MUST** be used to understand a file's existing implementation, conventions, and context before you suggest any edits.
*   **`FindFiles`:** Use to locate relevant modules or discover the project structure.
*   **`Edit`:** Apply changes surgically. Do not rewrite entire files.
*   **`Shell`:** Use for all command-line operations. This is **REQUIRED** for running `pre-commit`.
    *   **Docker Multi-Stage Builds:** When building Docker images with multiple stages, use the `--target` flag to specify the desired build stage (e.g., `docker build --target cpu .`).

## 11. Directory- and File-Specific Instructions

*   #### `on_fire_actions/`
    *   **Rule:** Any new notification service **MUST** be implemented in its own file as a class with an `async def __call__(self)` method.
    *   **Rule:** Configuration **MUST** be passed to the class's `__init__`. The class must not read from the environment itself.

*   #### `fire_detection/`
    *   **`base_fire_detection/utils.py`:** The functions `_detect_fire` and `_cuda_detect_fire` are performance-critical. Prioritize efficiency.

*   #### `stream_recording/`
    *   **Rule:** The `AsyncVideoChunkSaver` class contains complex logic. **DO NOT** alter its core state machine unless requested.

*   #### `burning_goat_detection.py`
    *   **Role:** This file is the central coordinator. **DO NOT** add business logic directly into this file.

## 12. Project Context Map

*   **Entry Point & Orchestration:** `burning_goat_detection.py`
*   **Core CV Logic:** `fire_detection/`
*   **Notifications:** `on_fire_actions/`
*   **Video Saving & Archiving:** `stream_recording/`
*   **Configuration Loading:** `setting.py`
*   **Dependencies:** `requirements.txt`, `pyproject.toml`
*   **Dev Tools & Linting:** `requirements_dev.txt`
