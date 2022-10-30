const schema = {
  $id: "https://github.com/MLIR-China/mlir-playground/blob/main/components/Sharing/schema/0.0.1.json",
  title: "MLIR Playground State",
  description: "Portable format for MLIR Playground state.",
  type: "object",
  properties: {
    version: {
      description: "The schema version this file follows.",
      type: "string",
      pattern: "^[0-9]+\\.[0-9]+\\.[0-9]+$",
    },
    environment: {
      description: "The compiler environment that this file is created with.",
      type: "string",
    },
    input: {
      description: "The initial input at the beginning of the pipeline.",
      type: "string",
    },
    stages: {
      description: "Information about each stage in the pipeline.",
      type: "array",
      minItems: 1,
      items: {
        type: "object",
        properties: {
          preset: {
            description: "The name of the preset being used for this stage.",
            type: "string",
          },
          arguments: {
            description:
              "The user-editable portion of the argument string that will be used to run the compiled program with.",
            type: "string",
          },
          editors: {
            description: "Contents of each of the editors.",
            type: "array",
            items: {
              type: "object",
              properties: {
                name: {
                  description: "Name of this editor tab.",
                  type: "string",
                },
                contents: {
                  description: "The full contents of this editor",
                  type: "string",
                },
              },
              required: ["name", "contents"],
            },
          },
          output: {
            description: "The initial content to fill the output window with.",
            type: "string",
          },
        },
        required: ["preset", "arguments", "editors"],
      },
    },
  },
  required: ["version", "stages"],
} as const;

export default schema;
