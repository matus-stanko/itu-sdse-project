package main

import ( 
    "context"
    "fmt"
    "log"
    "os"

    "dagger.io/dagger"
)

func main() {
    ctx := context.Background()

    // Connects to Dagger engine
    client, err := dagger.Connect(ctx, dagger.WithLogOutput(os.Stdout))
    if err != nil { // If connection fails, exit
        log.Fatalf("failed to connect to Dagger: %v", err)
    }
    defer client.Close() // Dagger connection closes when main ends

    // Root of the repo is a level higher than go/ dir
    // Mounts the root into the pipeline
    src := client.Host().Directory("..")

    // Python container
    py := client.Container().
        From("python:3.12-slim"). // Create a new container based on python 3.12
        WithDirectory("/src", src). // Mount repo root from host into /src inside container
        WithWorkdir("/src").        // Sets /src as working directory for commands
        WithEnvVariable("PIP_DISABLE_PIP_VERSION_CHECK", "1") // Disables pip's warnings

    // Upgrade pip to latest ver and install all requirements + dvc
    py = py.
        WithExec([]string{"python", "-m", "pip", "install", "--upgrade", "pip"}).
        WithExec([]string{"pip", "install", "-r", "requirements.txt", "dvc"})

    // Start DVC and update, and then run all the steps
    py = py.
        WithExec([]string{"dvc", "update", "data/raw/raw_data.csv.dvc"}).
        WithExec([]string{"python", "itu_sdse_project/dataset.py"}).
        WithExec([]string{"python", "itu_sdse_project/features.py"}).
        WithExec([]string{"python", "itu_sdse_project/modeling/train.py"})

    // Model in /src/model/model.pkl
    modelFile := py.File("/src/model/model.pkl")

    // Copies model file from container back to the host machine at ../model/model.pkl
    if _, err := modelFile.Export(ctx, "../model/model.pkl"); err != nil {
        log.Fatalf("failed to export model: %v", err)
    }

    fmt.Println("Dagger pipeline finished, model saved to root/model/model.pkl")
}