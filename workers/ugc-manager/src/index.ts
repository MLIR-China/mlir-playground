import { validateAgainstSchema } from "../shared/schema/validation";

const USER_CONTENT_SIZE_LIMIT = 128 * 2 ** 10; // 128KB

function createUserErrorResponse(message: string) {
  return new Response(JSON.stringify({ error: message }), {
    status: 400,
  });
}

async function handleRequest(request: Request, env: any) {
  /// Legalize incoming data.
  const readResult: { success: boolean; result: string } = await request
    .text()
    .then(
      (result) => {
        return { success: true, result: result };
      },
      (failure) => {
        return { success: false, result: failure };
      }
    );

  // Check: body can be read as string.
  if (!readResult.success) {
    return createUserErrorResponse(
      "Failed to parse read incoming data: " + readResult.result
    );
  }

  // Check: size is within limit
  const encoder = new TextEncoder();
  const encodedData = encoder.encode(readResult.result);
  if (encodedData.length > USER_CONTENT_SIZE_LIMIT) {
    return createUserErrorResponse("Data size exceeds limit.");
  }

  // Check: body is JSON.
  let parsedResult = null;
  try {
    parsedResult = JSON.parse(readResult.result);
  } catch (error: any) {
    return createUserErrorResponse(
      "Failed to parse incoming data as JSON." + error.message
    );
  }

  // Check: data conforms to latest schema
  let errorMsg = validateAgainstSchema(parsedResult);
  if (errorMsg) {
    return createUserErrorResponse(
      "Incoming data does not conform to latest schema: " + errorMsg
    );
  }

  /// Everything legal. Save into R2.
  const hash = await crypto.subtle.digest("SHA-256", encodedData);
  const filename = hash + ".json";
  // await env.UGC_BUCKET.put(filename, readResult.result);

  return new Response(
    JSON.stringify({
      filename: filename,
    })
  );
}

export default {
  async fetch(request: Request, env: any) {
    return await handleRequest(request, env);
  },
};
