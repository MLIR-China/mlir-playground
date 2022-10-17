import { expect, test } from "@jest/globals";

import { importFromSchema, validateAgainstSchema } from "./ImportExport";

test("catch empty input", () => {
  expect(validateAgainstSchema("")).toBeTruthy();
});

test("catch non JSON input", () => {
  expect(validateAgainstSchema("foobar")).toBeTruthy();
});

test("accept valid input", () => {
  const inputJson = {
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

  expect(validateAgainstSchema(inputJson)).toHaveLength(0);
  expect(importFromSchema(inputJson)).toMatchObject({
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
  });
});
