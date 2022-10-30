import { Validator, Schema } from "@cfworker/json-schema";

import Schema_0_0_1 from "./raw/0.0.1.json";

const validator = new Validator(Schema_0_0_1 as Schema);

// Returns an error message. If empty, means validation passed.
export function validateAgainstSchema(source: any): string {
  const validationResult = validator.validate(source);
  if (validationResult.valid) {
    return "";
  }
  return validationResult.errors[0].error;
}
