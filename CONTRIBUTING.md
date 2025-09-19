# Contributing to the train-llama docker image

Thank you for considering contributing to the train-llama docker image! I appreciate your interest in improving this project. To keep things running smoothly, please follow the guidelines below when contributing.

## How to Contribute

### 1. Fork the Repository

If you'd like to contribute to the project, start by forking this repository to your own GitHub account.

### 2. Clone Your Fork

Clone the forked repository to your local machine:

```bash
git clone https://github.com/soos/train-llama.git
cd train-llama
```

### 3. Create a New Branch

Create a new branch for your feature or bugfix or whatever:

```bash
git checkout -b branchy-the-branch
```

### 4. Make Your Changes

Make your changes, whether it's fixing bugs, improving documentation, or adding new features.
Make sure your changes don't break anything. Be especially careful when changing [the dockerfile][./Dockerfile]

### 5. Commit Your Changes

When committing, please follow these guidelines:

- Write clear, concise commit messages that adhere to the [commit message guidelines](#commit-message-guidelines).
- Confirm your changes won't break anything. Be especially careful if you changed [the dockerfile][./Dockerfile]

### 6. Push Your Changes

Push your changes to your forked repository:

```bash
git push origin branchy-the-branch
```

### 7. Create a Pull Request

Once your changes are pushed, create a pull request to the main repository. Ensure that your pull request is well described and linked to any relevant issues.

## Commit Message Guidelines

For clarity and ease of understanding, follow these guidelines for commit messages:

- **Use Present Tense**: Describe your change as if you're currently making it (e.g., "Fix Docker build error on Windows machines").
- **Reference Issues**: If your commit addresses a specific issue, reference it in the message.

## Reporting Issues

If you encounter any bugs or issues, and you don't know how to fix them with a pull request, please create a new issue in the repository and provide as much detail as possible (steps to reproduce, error messages, etc.).
