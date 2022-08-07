import React, { useRef, useState } from "react";
import type { NextPage } from "next";
import Head from "next/head";

import {
  Box,
  Button,
  Flex,
  Grid,
  GridItem,
  Heading,
  HStack,
  Image,
  Input,
  InputGroup,
  InputLeftAddon,
  InputRightAddon,
  Select,
  Spacer,
  Text,
  Textarea,
  VStack,
} from "@chakra-ui/react";
import Editor, { OnMount } from "@monaco-editor/react";
import styles from "../styles/Home.module.css";

import {
  defaultFacility,
  getFacilityNames,
  getFacility,
} from "../components/Facilities/FacilitySelector";

import NavBar from "../components/UI/navbar";

const Home: NextPage = () => {
  const monospaceFontFamily = "Consolas, 'Courier New', monospace";

  // state
  const [allEditorsMounted, setAllEditorsMounted] = useState(false);
  const cppEditor: React.MutableRefObject<any> = useRef(null);
  const inputEditor: React.MutableRefObject<any> = useRef(null);
  const outputEditor: React.MutableRefObject<any> = useRef(null);
  const [logValue, setLogValue] = useState<Array<string>>([]);

  const [currentFacility, setCurrentFacility] = React.useState(defaultFacility);

  const [runArgsLeftAddon, setRunArgsLeftAddon] = useState("");
  const [runArgsRightAddon, setRunArgsRightAddon] = useState("");
  const [additionalRunArgs, setAdditionalRunArgs] = useState("");
  const [inputEditorFileName, setInputEditorFileName] = useState("");
  const [outputEditorFileName, setOutputEditorFileName] = useState("");

  const [runStatus, setRunStatus] = useState("");

  function setFacilitySelection(selection: string) {
    setCurrentFacility(selection);
    const props = getFacility(selection);
    cppEditor.current.updateOptions({
      readOnly: !props.isCodeEditorEnabled(),
    });
    setRunArgsLeftAddon(props.getRunArgsLeftAddon());
    setRunArgsRightAddon(props.getRunArgsRightAddon());
    setAdditionalRunArgs(props.getDefaultAdditionalRunArgs());
    setInputEditorFileName(props.getInputFileName());
    setOutputEditorFileName(props.getOutputFileName());
    cppEditor.current.setValue(props.getDefaultCodeFile());
    inputEditor.current.setValue(props.getDefaultInputFile());
    outputEditor.current.setValue("");
  }

  const onEditorMounted = (editorRef: React.MutableRefObject<any>): OnMount => {
    return (editor, _) => {
      editorRef.current = editor;
      if (cppEditor.current && inputEditor.current && outputEditor.current) {
        // All editors mounted.
        window.addEventListener("resize", () => {
          cppEditor.current.layout({});
          inputEditor.current.layout({});
          outputEditor.current.layout({});
        });
        setAllEditorsMounted(true);
        setFacilitySelection(defaultFacility);
      }
    };
  };

  const onFacilitySelectionChange = (
    event: React.ChangeEvent<HTMLSelectElement>
  ) => {
    setFacilitySelection(event.target.value);
  };

  const onRunButtonClick = () => {
    const input_mlir = inputEditor.current.getValue();
    setLogValue((currValue) => [...currValue, ""]);
    const printer = (text: string) => {
      setLogValue((currValue) => [
        ...currValue.slice(0, -1),
        currValue[currValue.length - 1] + text + "\n",
      ]);
    };

    const facility = getFacility(currentFacility);
    let cpp_source = "";
    if (facility.isCodeEditorEnabled()) {
      cpp_source = cppEditor.current.getValue();
      setRunStatus("Compiling...");
    } else {
      setRunStatus("Running...");
    }
    facility
      .run(cpp_source, input_mlir, additionalRunArgs, printer)
      .finally(() => {
        setRunStatus("");
      })
      .then((output: string) => {
        outputEditor.current.setValue(output);
      }, printer);
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

  const codeEditor = (
    <Editor
      key="cppEditor"
      height="100%"
      defaultLanguage="cpp"
      onMount={onEditorMounted(cppEditor)}
      options={monacoOptions}
    />
  );

  const inputMLIRViewer = (
    <Editor
      key="inputEditor"
      height="100%"
      defaultLanguage="cpp"
      onMount={onEditorMounted(inputEditor)}
      options={monacoOptions}
    />
  );

  const outputMLIRViewer = (
    <Editor
      key="outputEditor"
      height="100%"
      defaultLanguage="cpp"
      onMount={onEditorMounted(outputEditor)}
      options={{ ...monacoOptions, readOnly: true }}
    />
  );

  return (
    <div className={styles.container}>
      <Head>
        <title>MLIR Playground</title>
        <meta
          name="description"
          content="Play with MLIR right in the browser."
        />
        <link
          rel="icon"
          type="image/png"
          sizes="32x32"
          href="/favicon/favicon-32x32.png"
        />
        <link
          rel="icon"
          type="image/png"
          sizes="16x16"
          href="/favicon/favicon-16x16.png"
        />
        <link
          rel="apple-touch-icon"
          sizes="180x180"
          href="/favicon/apple-touch-icon.png"
        />
      </Head>
      <NavBar />
      <Flex
        as="main"
        direction="row"
        justify="space-between"
        className={styles.playground_flexbox}
        height="90vh"
        padding="1rem"
      >
        <Box height="100%" className={styles.main_left}>
          <VStack spacing={4} align="left" height="100%">
            <HStack>
              <Select
                value={currentFacility}
                onChange={onFacilitySelectionChange}
                disabled={!allEditorsMounted}
              >
                {getFacilityNames().map((name, i) => {
                  return (
                    <option value={name} key={i}>
                      {name}
                    </option>
                  );
                })}
              </Select>
              <Button
                isLoading={!allEditorsMounted || runStatus !== ""}
                mt="8"
                as="a"
                size="lg"
                colorScheme="blue"
                fontWeight="bold"
                onClick={onRunButtonClick}
              >
                Run
              </Button>
              <Text w="6rem">{runStatus || "Ready"}</Text>
            </HStack>
            <HStack>
              <Text>Arguments</Text>
              <InputGroup fontFamily={monospaceFontFamily}>
                <InputLeftAddon>{runArgsLeftAddon}</InputLeftAddon>
                <Input
                  value={additionalRunArgs}
                  onChange={(event) =>
                    setAdditionalRunArgs(event.target.value)
                  }
                ></Input>
                <InputRightAddon>{runArgsRightAddon}</InputRightAddon>
              </InputGroup>
            </HStack>
            <Flex height="80vh" flexDirection="column">
              <Flex align="end">
                <Heading>Editor</Heading>
                <Spacer />
                <Text fontFamily={monospaceFontFamily}>mlir-opt.cpp</Text>
              </Flex>
              <Box borderWidth="2px" flexGrow="1" h="100%">
                {codeEditor}
              </Box>
            </Flex>
          </VStack>
        </Box>
        <Flex height="100%" flexDirection="column" className={styles.main_right}>
          <Flex height="30vh" flexDirection="column">
            <Flex align="end">
              <Heading>Input</Heading>
              <Spacer />
              <Text fontFamily={monospaceFontFamily}>
                {inputEditorFileName}
              </Text>
            </Flex>
            <Box borderWidth="2px" flexGrow="1">
              {inputMLIRViewer}
            </Box>
          </Flex>
          <Flex minHeight="30vh" flexGrow="1" flexDirection="column">
            <Heading>Logs</Heading>
            <Box
              borderWidth="2px"
              flexGrow="1"
              height="100%"
              bg="gray.800"
              fontFamily={monospaceFontFamily}
              overflowY="auto"
              padding="4"
            >
              {logValue.map((logText, logIndex) => (
                <Box className={styles.log_content} key={logIndex}>
                  {logText}
                </Box>
              ))}
            </Box>
          </Flex>
          <Flex height="30vh" flexDirection="column">
            <Flex align="end">
              <Heading>Output</Heading>
              <Spacer />
              <Text fontFamily={monospaceFontFamily}>
                {outputEditorFileName}
              </Text>
            </Flex>
            <Box borderWidth="2px" flexGrow="1">
              {outputMLIRViewer}
            </Box>
          </Flex>
        </Flex>
      </Flex>
    </div>
  );
};

export default Home;
