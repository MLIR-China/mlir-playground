import { promises as fsPromises } from "fs";

import { compile, JSONSchema } from "json-schema-to-typescript";

import Schema_0_0_1 from "./workers/ugc-manager/shared/schema/versions/0.0.1.json";

compile(Schema_0_0_1 as JSONSchema, "Schema_0_0_1", {
  additionalProperties: false,
}).then((tsCode) =>
  fsPromises.writeFile(
    "./workers/ugc-manager/shared/schema/types/0.0.1.d.ts",
    tsCode
  )
);
