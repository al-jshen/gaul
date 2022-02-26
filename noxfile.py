import os
import tempfile

import nox

locations = "gaul", "tests", "noxfile.py", "docs/source/conf.py"
nox.options.sessions = "lint", "tests", "docstrings"
versions = ["3.9"]


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


@nox.session(python=versions)
def tests(session):
    args = session.posargs or ["--cov", "-m", "not e2e"]
    install_with_constraints(session, "coverage[toml]", "pytest", "pytest-cov")
    session.run("pytest", *args)


@nox.session(python=versions)
def lint(session):
    args = session.posargs or locations
    install_with_constraints(
        session,
        "flake8",
        "flake8-black",
        "flake8-bugbear",
    )
    session.run("flake8", *args)


@nox.session(python=versions)
def format(session):
    args = session.posargs or locations
    install_with_constraints(session, "isort", "black")
    session.run("isort", ".")
    session.run("black", *args)


@nox.session(python=versions)
def docstrings(session):
    def search_directories_for_python_files(directories):
        results = []
        for loc in directories:
            if os.path.isdir(loc):
                for file in os.listdir(loc):
                    if file.endswith(".py"):
                        results.append(os.path.join(loc, file))
            else:
                if os.path.isfile(loc) and loc.endswith(".py"):
                    results.append(loc)
        return results

    # have any arguments?
    if session.posargs:
        # the only argument is --in-place?
        if len(session.posargs) == 1 and session.posargs[0] == "--in-place":
            # yes, so we'll search for python files in the current directory
            # and format them in-place
            args = ["--in-place"]
            args.extend(search_directories_for_python_files(locations))
        else:
            # no, which means there are arguments/files but we don't want
            # to format them in-place
            args = search_directories_for_python_files(session.posargs)
    else:
        # no arguments, so we'll run the docstring checker on the whole project
        # and not format anything in-place
        args = search_directories_for_python_files(locations)

    install_with_constraints(session, "docformatter")
    session.run(
        "docformatter",
        "--pre-summary-newline",
        "--make-summary-multi-line",
        *args,
    )


@nox.session(python=versions)
def coverage(session):
    install_with_constraints(session, "coverage[toml]", "codecov")
    session.run("coverage", "xml", "--fail-under=0")
    session.run("codecov", *session.posargs)


@nox.session(python=versions)
def docs(session):
    """
    Build the docs for this package.
    """
    install_with_constraints(
        session, "sphinx", "sphinx-book-theme", "sphinx-autodoc-typehints"
    )
    session.run("sphinx-build", "-b", "html", "docs/source", "docs/build/html")


@nox.session(python=versions)
def autodoc(session):
    """
    Run sphinx-autodoc to generate documentation from docstrings.
    """
    install_with_constraints(session, "sphinx", "sphinx-autodoc-typehints")
    session.run("sphinx-apidoc", "-o", "docs/source/user", "gaul")
