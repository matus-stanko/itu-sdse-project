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

	// pripojenie k Dagger engine
	client, err := dagger.Connect(ctx, dagger.WithLogOutput(os.Stdout))
	if err != nil {
		log.Fatalf("failed to connect to Dagger: %v", err)
	}
	defer client.Close()

	// root tvojho repa na hoste je o level vyššie než go/
	// go/           -> tu beží tento Go kód
	// .. (= /src)   -> root: tu je requirements.txt, data/, itu_sdse_project/, model/
	src := client.Host().Directory("..")

	// Python container
	py := client.Container().
		From("python:3.12-slim").
		WithDirectory("/src", src). // mountne celý repo root do /src
		WithWorkdir("/src").        // všetko sa spúšťa z rootu
		WithEnvVariable("PIP_DISABLE_PIP_VERSION_CHECK", "1")

	// nainštaluj dependencies
	py = py.
		WithExec([]string{"python", "-m", "pip", "install", "--upgrade", "pip"}).
		WithExec([]string{"pip", "install", "-r", "requirements.txt", "dvc"})

	// spusti DVC + tvoje kroky
	py = py.
		WithExec([]string{"dvc", "update", "data/raw/raw_data.csv.dvc"}).
		WithExec([]string{"python", "itu_sdse_project/dataset.py"}).
		WithExec([]string{"python", "itu_sdse_project/features.py"}).
		WithExec([]string{"python", "itu_sdse_project/modeling/train.py"})

	// model by mal byť vytvorený v /src/model/model.pkl
	modelFile := py.File("/src/model/model.pkl")

	// exportni ho do root/model/model.pkl
	// z pohľadu priečinka `go/` je root = ".."
	if _, err := modelFile.Export(ctx, "../model/model.pkl"); err != nil {
		log.Fatalf("failed to export model: %v", err)
	}

	fmt.Println("Dagger pipeline finished, model saved to root/model/model.pkl")
}
