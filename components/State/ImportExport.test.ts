import { expect, test } from "@jest/globals";

import { importFromSchema, SchemaObjectType } from "../State/ImportExport";

test("accept valid input", () => {
  const inputJson: SchemaObjectType = {
    version: "0.0.1",
    environment: "myenv",
    input: "// test",
    stages: [
      {
        preset: "TableGen DRR",
        arguments: "",
        editors: [
          {
            name: "DRR",
            contents: "blah",
          },
          {
            name: "Generated",
            contents: "blah",
          },
          {
            name: "Driver",
            contents: "blah",
          },
        ],
      },
    ],
  };

  expect(importFromSchema(inputJson)).toMatchObject({
    ok: true,
    value: {
      input: inputJson.input,
      stages: [
        {
          preset: inputJson.stages[0].preset,
          additionalRunArgs: inputJson.stages[0].arguments,
          editorContents: inputJson.stages[0].editors.map(
            (editor) => editor.contents
          ),
          currentPaneIdx: 0,
          output: "",
        },
      ],
    },
  });
});
