/**
 * Set allow CORS on Cloudflare R2 buckets.
 *
 * Requires npm package @aws-sdk/client-s3.
 */

const R2_ENDPOINT = "<FILL IN R2 ENDPOINT HERE>";

import {
  S3Client,
  GetBucketCorsCommand,
  PutBucketCorsCommand,
} from "@aws-sdk/client-s3";

const client = new S3Client({
  region: "auto",
  endpoint: R2_ENDPOINT,
});

const corsRules = [
  {
    AllowedHeaders: ["*"],
    AllowedMethods: ["GET"],
    AllowedOrigins: ["*"],
    ExposeHeaders: [],
    MaxAgeSeconds: 3000,
  },
];

const putCommand = new PutBucketCorsCommand({
  Bucket: "mlir-playground-ugc",
  CORSConfiguration: { CORSRules: corsRules },
});
client.send(putCommand).then((response) => {
  console.log("PUT response:\n" + response);

  // Get and check
  const getCommand = new GetBucketCorsCommand({
    Bucket: "mlir-playground-ugc",
  });
  client.send(getCommand).then((response) => {
    console.log("GET response:\n" + response);
  });
});
