import React, { useRef, useState } from "react";
import type { NextPage } from "next";
import Head from "next/head";

import {
  Box,
  Divider,
  Flex,
  Heading,
  HStack,
  Input,
  InputGroup,
  InputLeftAddon,
  InputRightAddon,
  Select,
  Text,
  VStack,
} from "@chakra-ui/react";
import { OnMount } from "@monaco-editor/react";
import styles from "../styles/Home.module.css";
import { monospaceFontFamily } from "../components/UI/constants";

import {
  defaultPreset,
  getPresetNames,
  getPreset,
} from "../components/Presets/PresetFactory";

import LabeledEditor from "../components/UI/labeledEditor";
import NavBar from "../components/UI/navbar";
import WasmCompiler from "../components/WasmCompiler";
import { RunStatus } from "../Utils/RunStatus";

const Home: NextPage = () => {
  // state
  const [allEditorsMounted, setAllEditorsMounted] = useState(false);
  const cppEditor: React.MutableRefObject<any> = useRef(null);
  const inputEditor: React.MutableRefObject<any> = useRef(null);
  const outputEditor: React.MutableRefObject<any> = useRef(null);
  const [logValue, setLogValue] = useState<Array<string>>([]);

  const [currentPreset, setCurrentPreset] = React.useState(defaultPreset);

  const [runArgsLeftAddon, setRunArgsLeftAddon] = useState("");
  const [runArgsRightAddon, setRunArgsRightAddon] = useState("");
  const [additionalRunArgs, setAdditionalRunArgs] = useState("");
  const [inputEditorFileName, setInputEditorFileName] = useState("");
  const [outputEditorFileName, setOutputEditorFileName] = useState("");

  const [compilerDataCached, setCompilerDataCached] = useState(false);
  const [runStatus, setRunStatus] = useState("");
  const [runProgress, setRunProgress] = useState(0);

  function updateCompilerDataCached() {
    WasmCompiler.dataFilesCached().then((isCached) => {
      setCompilerDataCached(isCached);
    });
  }

  function setPresetSelection(selection: string) {
    setCurrentPreset(selection);
    const props = getPreset(selection);
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
        setPresetSelection(defaultPreset);
        updateCompilerDataCached();
      }
    };
  };

  const onPresetSelectionChange = (
    event: React.ChangeEvent<HTMLSelectElement>
  ) => {
    setPresetSelection(event.target.value);
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
    const statusListener = (status: RunStatus) => {
      setRunStatus(status.label);
      setRunProgress(status.percentage);
      updateCompilerDataCached();
    };

    const preset = getPreset(currentPreset);
    let cpp_source = "";
    if (preset.isCodeEditorEnabled()) {
      cpp_source = cppEditor.current.getValue();
      setRunStatus("Compiling...");
    } else {
      setRunStatus("Running...");
    }
    preset
      .run(cpp_source, input_mlir, additionalRunArgs, printer, statusListener)
      .finally(() => {
        setRunStatus("");
        setRunProgress(100);
      })
      .then((output: string) => {
        outputEditor.current.setValue(output);
      }, printer);
  };

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
      <NavBar
        allEditorsMounted={allEditorsMounted}
        localEnvironmentReady={compilerDataCached}
        runStatus={runStatus}
        runProgress={runProgress}
        onClick={onRunButtonClick}
      />
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
            <PresetSelector
              preset={currentPreset}
              onPresetChange={onPresetSelectionChange}
              disabled={!allEditorsMounted}
            />
            <ArgumentsBar
              leftAddon={runArgsLeftAddon}
              rightAddon={runArgsRightAddon}
              additionalRunArgs={additionalRunArgs}
              setAdditionalRunArgs={setAdditionalRunArgs}
            />
            <LabeledEditor
              height="80vh"
              label="Editor"
              filename="mlir-opt.cpp"
              onMount={onEditorMounted(cppEditor)}
            />
          </VStack>
        </Box>
        <Divider orientation="vertical" />
        <Flex
          height="100%"
          flexDirection="column"
          className={styles.main_right}
        >
          <LabeledEditor
            height="30vh"
            label="Input"
            filename={inputEditorFileName}
            onMount={onEditorMounted(inputEditor)}
          />
          <TransformationOutput
            logWindowProps={{ height: "30vh", logs: logValue }}
            labeledEditorProps={{
              height: "30vh",
              label: "Output",
              filename: outputEditorFileName,
              onMount: onEditorMounted(outputEditor),
            }}
          />
        </Flex>
      </Flex>
    </div>
  );
};

type TransformationOutputProps = {
  logWindowProps: Parameters<typeof LogWindow>[0];
  labeledEditorProps: Parameters<typeof LabeledEditor>[0];
};

const TransformationOutput = (props: TransformationOutputProps) => {
  return (
    <Flex flexDirection="column">
      <HStack height="30vh">
        <svg
          xmlns="http://www.w3.org/2000/svg"
          height="30vh"
          width="20px"
          className={styles.log_arrow}
        >
          <g className={styles.log_arrow_g} strokeWidth="2" stroke="lightgray">
            <line x1="-10" y1="-2" x2="-10" y2="-100%" />
            <line x1="-10" y1="-2" x2="-2" y2="-16" />
            <line x1="-10" y1="-2" x2="-18" y2="-16" />
          </g>
        </svg>
        <LogWindow {...props.logWindowProps} />
      </HStack>
      <LabeledEditor {...props.labeledEditorProps} />
    </Flex>
  );
};

type LogWindowProps = {
  height: string;
  logs: Array<String>;
};

const LogWindow = (props: LogWindowProps) => {
  return (
    <Flex height={props.height} flexGrow="1" flexDirection="column">
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
        {props.logs.map((logText, logIndex) => (
          <Box className={styles.log_content} key={logIndex}>
            {logText}
          </Box>
        ))}
      </Box>
    </Flex>
  );
};

type PresetSelectorProps = {
  preset: string;
  onPresetChange: (event: React.ChangeEvent<HTMLSelectElement>) => void;
  disabled: boolean;
};

const PresetSelector = (props: PresetSelectorProps) => {
  return (
    <HStack>
      <Text>Preset</Text>
      <Select
        value={props.preset}
        onChange={props.onPresetChange}
        disabled={props.disabled}
      >
        {getPresetNames().map((name, i) => {
          return (
            <option value={name} key={i}>
              {name}
            </option>
          );
        })}
      </Select>
    </HStack>
  );
};

type ArgumentsBarProps = {
  leftAddon: string;
  rightAddon: string;
  additionalRunArgs: string;
  setAdditionalRunArgs: (text: string) => void;
};

const ArgumentsBar = (props: ArgumentsBarProps) => {
  return (
    <HStack>
      <Text>Arguments</Text>
      <InputGroup fontFamily={monospaceFontFamily}>
        <InputLeftAddon>{props.leftAddon}</InputLeftAddon>
        <Input
          value={props.additionalRunArgs}
          onChange={(event) => props.setAdditionalRunArgs(event.target.value)}
        ></Input>
        <InputRightAddon>{props.rightAddon}</InputRightAddon>
      </InputGroup>
    </HStack>
  );
};

export default Home;
