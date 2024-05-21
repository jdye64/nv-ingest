# Continuous Integration Directory

## Introduction

This directory contains all scripts, configurations, and tools related to Continuous Integration (CI) for the project. It includes various build scripts, configuration files, and utilities required for automating the build, test, and deployment processes.

## Table of Contents

- [Introduction](#introduction)
- [Building and Publishing Packages](#building-and-publishing-packages)
  - [Usage](#usage)
  - [Parameters](#parameters)
  - [Examples](#examples)
  - [Output](#output)
  - [Installation](#installation)
  - [Additional Notes](#additional-notes)

## Building and Publishing Packages

This project includes a build script to handle the creation of Python packages for both the client and service libraries. The script is located at `./ci/scripts/build_pip_packages.sh` and supports both dev and release builds.

### Usage

To use the build script, you need to specify the type of build (`dev` or `release`) and the library (`client` or `service`). The script will automatically set the version based on the current date and build type, then build the corresponding package.

#### Command Syntax

```sh
./ci/scripts/build_pip_packages.sh --type <dev|release> --lib <client|service>
```

#### Parameters

- `--type <dev|release>`: Specifies the type of build.
  - `dev`: Creates a development build with a version suffix `-dev`.
  - `release`: Creates a release build with the current date as the version.
- `--lib <client|service>`: Specifies the library to build.
  - `client`: Builds the client library.
  - `service`: Builds the service library.

#### Examples

1. **Nightly Build for Client Library**

   ```sh
   ./ci/scripts/build_pip_packages.sh --type dev --lib client
   ```

   This command will create a development build of the client library with a version format `YYYY.MM.DD-dev`.

2. **Release Build for Service Library**

   ```sh
   ./ci/scripts/build_pip_packages.sh --type release --lib service
   ```

   This command will create a release build of the service library with a version format `YYYY.MM.DD`.

### Output

The script will generate the distribution files in the `dist/` directory of the respective library's root directory. For example, running the script for the client library will create the distribution files in `client/dist/`.

### Installation

After building the package, you can install it using `pip`:

```sh
pip install path/to/dist/package_name-version-py3-none-any.whl
```

Replace `path/to/dist/package_name-version-py3-none-any.whl` with the actual path to the built wheel file.

### Additional Notes

- Ensure that you have the necessary permissions and environment variables set up for building and installing the packages.
- The build script must be executable. If it's not, you can make it executable with the following command:

  ```sh
  chmod +x ./ci/scripts/build_pip_packages.sh
  ```

By following these instructions, you can efficiently build and manage both dev and release packages for the client and service libraries.
