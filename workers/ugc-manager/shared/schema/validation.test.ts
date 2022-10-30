import { expect, test } from "@jest/globals";

import { validateAgainstSchema } from "./validation";

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
});
