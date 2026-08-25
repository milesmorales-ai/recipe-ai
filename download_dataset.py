from datasets import load_dataset
import pandas as pd
from pathlib import Path

print("Downloading RecipeNLG...")

recipe_limit = 200_000
output_name = f"recipes_{recipe_limit}.csv"
project_root = Path(__file__).parent
output_paths = [
    project_root / output_name,
    project_root / "app" / "public" / output_name,
]

dataset = load_dataset(
    "Mahimas/recipenlg",
    split="train",
    streaming=True,
)

# Take a larger slice from the streamed dataset so the app stays responsive.
recipes = list(dataset.take(recipe_limit))

df = pd.DataFrame(recipes)
for output_path in output_paths:
    df.to_csv(output_path, index=False, encoding="utf-8")

print("Done!")
print(f"Saved {len(df)} recipes.")
print(f"Files: {', '.join(str(path) for path in output_paths)}")
