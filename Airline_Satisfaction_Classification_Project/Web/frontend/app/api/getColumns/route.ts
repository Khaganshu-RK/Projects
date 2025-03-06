import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";
import yaml from "js-yaml";

export async function GET() {
  try {
    // Get absolute path to YAML file
    const yamlFilePath = path.join(
      process.cwd(),
      "app",
      "data",
      "data_schema.yaml"
    );

    // Debug: Print file path
    console.log("Loading YAML from:", yamlFilePath);

    // Check if file exists
    if (!fs.existsSync(yamlFilePath)) {
      console.error("YAML file not found!");
      return NextResponse.json(
        { success: false, error: "YAML file not found" },
        { status: 404 }
      );
    }

    // Read the YAML file
    const fileContents = fs.readFileSync(yamlFilePath, "utf8");

    // Debug: Print file contents
    // console.log("YAML File Contents:", fileContents);

    // Parse YAML into JavaScript object
    const schema = yaml.load(fileContents);

    // Return response
    return NextResponse.json({ success: true, schema });
  } catch (error) {
    console.error("Error parsing YAML:", error);
    return NextResponse.json(
      { success: false, error: "Invalid YAML format" },
      { status: 500 }
    );
  }
}
