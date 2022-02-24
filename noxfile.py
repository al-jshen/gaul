import tempfile

import nox

locations = "jacket", "tests", "noxfile.py"
nox.options.sessions = "lint", "tests"


def install_with_constraints(session, *args, **kwargs):
    with tempfile.NamedTemporaryFile() as reqs:
        session.run(
            "poetry",
            "export",
            "--dev",
            "--format=requirements.txt",
            "--without-hashes",
            f"--output={reqs.name}",
            external=True,
        )
        session.install(f"--constraint={reqs.name}", *args, **kwargs)


@nox.session(python=["3.9"])
def tests(session):
    args = session.posargs or ["--cov", "-m", "not e2e"]
    session.run("poetry", "install", "--no-dev", external=True)
    install_with_constraints(session, "coverage[toml]", "pytest", "pytest-cov")
    session.run("pytest", *args)


@nox.session(python=["3.9"])
def lint(session):
    args = session.posargs or locations
    install_with_constraints(
        session, "flake8", "flake8-black", "flake8-import-order", "flake8-bugbear"
    )
    session.run("flake8", *args)


@nox.session(python=["3.9"])
def black(session):
    args = session.posargs or locations
    install_with_constraints(session, "black")
    session.run("black", *args)
