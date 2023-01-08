/**
 * Set allow CORS on Cloudflare R2 buckets.
 *
 * Requires npm package @aws-sdk/client-s3.
 */

import {
  S3Client,
  GetBucketCorsCommand,
  PutBucketCorsCommand,
} from "@aws-sdk/client-s3";

if (process.argv.length != 4) {
  console.error("Expected two command line arguments.");
  console.log("Usage: npx ts-node set_r2_cors.ts <R2_ENDPOINT> <BUCKET_NAME>");
  process.exit(1);
}

// First 2 args are "ts-node" and the script file respectively.
const R2_ENDPOINT = process.argv[2];
const BUCKET_NAME = process.argv[3];

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
  Bucket: BUCKET_NAME,
  CORSConfiguration: { CORSRules: corsRules },
});
client.send(putCommand).then((response) => {
  console.log("PUT response:\n" + response);

  // Get and check
  const getCommand = new GetBucketCorsCommand({
    Bucket: BUCKET_NAME,
  });
  client.send(getCommand).then((response) => {
    console.log("GET response:\n" + response);
  });
});
