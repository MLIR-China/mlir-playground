import React from "react";
import Editor, { OnMount } from "@monaco-editor/react";
import { Box, Flex, Heading, Spacer, Text } from "@chakra-ui/react";

import { monospaceFontFamily } from "./constants";


type LabeledEditorProps = {
  height: string,
  label: string,
  filename: string,
  onMount: OnMount,
};

const monacoOptions = {
  selectOnLineNumbers: true,
  quickSuggestions: true,
  minimap: {
    enabled: false,
  },
  scrollBeyondLastLine: false,
  automaticLayout: true,
};

const LabeledEditor = (props: LabeledEditorProps) => {
  return (
    <Flex height={props.height} flexDirection="column">
      <Flex align="end">
        <Heading>{props.label}</Heading>
        <Spacer />
        <Text fontFamily={monospaceFontFamily}>{props.filename}</Text>
      </Flex>
      <Box borderWidth="2px" flexGrow="1" h="100%">
        <Editor
          height="100%"
          defaultLanguage="cpp"
          onMount={props.onMount}
          options={monacoOptions}
        />
      </Box>
    </Flex>
  );
};

export default LabeledEditor;
