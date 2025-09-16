# Contributing to "Is the Gävle Goat Burning?"

Thank you for your interest in contributing! Adhering to these guidelines ensures a smooth and effective development process for everyone.

## Code of Conduct

This project and everyone participating in it is governed by a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Development Workflow

1.  **Fork & Clone:** Fork the repository and clone it to your local machine.
2.  **Branch:** Create a new branch for your feature or bug fix: `git checkout -b <type>/<short-description>`.
    -   *Examples:* `feat/add-slack-notifications`, `fix/email-auth-error`
3.  **Code:** Make your changes. See the "Coding Standards" section below.
4.  **Test:** Run the test suite to ensure your changes don't break existing functionality.
5.  **Commit:** Follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) standard for your commit messages.
6.  **Push & PR:** Push your branch to your fork and open a pull request to the `main` branch of the original repository.

## Coding Standards

### 1. Code Style and Linting

We use `ruff` for linting and formatting, enforced via `pre-commit`.

-   **Setup:**
    ```bash
    pip install -r requirements-dev.txt
    pre-commit install
    ```
-   **Validation:** Before committing, always run the checks:
    ```bash
    pre-commit run --all-files
    ```

### 2. Type Hinting

-   **Requirement:** All function and method signatures **MUST** include type hints for all arguments and return values.
-   **Clarity:** Use the most specific types possible (e.g., `list[str]` instead of `list`).
-   **Complex Types:** For complex or reusable type definitions, use `typing.TypeAlias`.

### 3. Docstrings

-   **Requirement:** All modules, classes, public methods, and functions **MUST** have a docstring.
-   **Format:** We use the **Google Style** docstring format.

-   **Example:**
    ```python
    """A brief one-line summary of the module's purpose.

    A more detailed description of the module's contents, architecture,
    and any other relevant information.
    """

    class ExampleClass:
        """A summary of the class's purpose.

        Attributes:
            attr1 (str): Description of the first attribute.
            attr2 (int): Description of the second attribute.
        """

        def example_method(self, param1: int, param2: str) -> bool:
            """Does something interesting.

            This method serves as an example of a complete docstring. It details
            the arguments, what the method returns, and any exceptions it might raise.

            Args:
                param1: The first parameter, an integer.
                param2: The second parameter, a string.

            Returns:
                True if the operation was successful, False otherwise.

            Raises:
                ValueError: If `param1` is negative.
            """
            if param1 < 0:
                raise ValueError("param1 cannot be negative")
            return True
    ```

### 4. Magic Numbers

-   **Rule:** Avoid using unnamed, "magic" numbers directly in the code.
-   **Implementation:** Define them as module-level or class-level constants with descriptive, `UPPER_SNAKE_CASE` names.
-   **Example:**
    ```python
    # Bad
    blur = cv2.GaussianBlur(frame, (21, 21), 0)

    # Good
    GAUSSIAN_BLUR_KERNEL_SIZE = (21, 21)
    blur = cv2.GaussianBlur(frame, GAUSSIAN_BLUR_KERNEL_SIZE, 0)
    ```

## Commit Messages

We follow the **Conventional Commits** specification. This helps automate changelogs and makes the project history easy to read.

-   **Format:** `<type>(<scope>): <description>`
-   **Common Types:**
    -   `feat`: A new feature.
    -   `fix`: A bug fix.
    -   `refactor`: A code change that neither fixes a bug nor adds a feature.
    -   `docs`: Documentation only changes.
    -   `test`: Adding missing tests or correcting existing tests.
    -   `chore`: Changes to the build process or auxiliary tools.

-   **Example Commit:**
    ```
    feat(notifications): add support for Slack webhooks

    This commit introduces a new `SendToSlack` class that allows sending
    fire notifications to a configured Slack channel.
    ```
