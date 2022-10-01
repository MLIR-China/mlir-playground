import React from "react";
import Editor, { OnChange, OnMount } from "@monaco-editor/react";
import { Box, Flex, Heading, Spacer, Text } from "@chakra-ui/react";

import { monospaceFontFamily } from "./constants";

type LabeledEditorProps = {
  height?: string;
  flex?: string;
  label?: string;
  filename: string;
  onChange: OnChange;
  onMount: OnMount;
  value: string;
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

const LabeledEditor = React.forwardRef(
  (props: LabeledEditorProps, ref: React.ForwardedRef<HTMLDivElement>) => {
    return (
      <Flex
        height={props.height}
        flex={props.flex}
        flexDirection="column"
        ref={ref}
        width="100%"
      >
        {props.label !== undefined && (
          <Flex align="end">
            <Heading>{props.label}</Heading>
            <Spacer />
            <Text fontFamily={monospaceFontFamily}>{props.filename}</Text>
          </Flex>
        )}
        <Box borderWidth="2px" flexGrow="1" h="100%">
          <Editor
            height="100%"
            defaultLanguage="cpp"
            path={props.filename}
            onChange={props.onChange}
            onMount={props.onMount}
            options={monacoOptions}
            value={props.value}
          />
        </Box>
      </Flex>
    );
  }
);
LabeledEditor.displayName = "LabeledEditor";

export default LabeledEditor;
