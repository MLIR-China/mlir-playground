import React, { useRef, useState } from "react";
import type { NextPage } from "next";
import Head from "next/head";

import {
  Box,
  Button,
  Divider,
  Flex,
  Heading,
  HStack,
  Input,
  InputGroup,
  InputLeftAddon,
  InputRightAddon,
  Select,
  Tag,
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
import { RunStatus } from "../components/Utils/RunStatus";

// Stores the configuration of a particular stage.
class StageState {
  preset: string;
  additionalRunArgs: string;
  editorContent: string;
  logs: Array<string>;
  output: string;

  constructor(preset: string = defaultPreset) {
    this.preset = preset;
    const presetProps = getPreset(preset);
    this.editorContent = presetProps.getDefaultCodeFile();
    this.additionalRunArgs = presetProps.getDefaultAdditionalRunArgs();
    this.logs = [];
    this.output = "";
  }
}

const Home: NextPage = () => {
  /* Compiler Environment Management */
  const [compilerEnvironmentVersion, setCompilerEnvironmentVersion] =
    useState("");
  const [compilerEnvironmentPopoverOpen, setCompilerEnvironmentPopoverOpen] =
    useState(false);
  const [runStatus, setRunStatus] = useState("");
  const [runProgress, setRunProgress] = useState(0);

  // Returns whether or not the data is cached after checking.
  function updateCompilerEnvironmentReady(): Promise<boolean> {
    return WasmCompiler.dataFilesCachedVersion().then((version) => {
      const isCached = !!version;
      setCompilerEnvironmentVersion(version);
      return isCached;
    });
  }

  function downloadCompilerEnvironment(): Promise<boolean> {
    if (compilerEnvironmentVersion) {
      return Promise.resolve(true);
    }

    return WasmCompiler.initialize().then((success) => {
      if (!success) {
        alert("Failed to initialize compiler environment.");
      }
      updateCompilerEnvironmentReady();

      return Promise.resolve(success);
    });
  }

  /* UI State */
  const [allEditorsMounted, setAllEditorsMounted] = useState(false);
  const cppEditor: React.MutableRefObject<any> = useRef(null);
  const inputEditor: React.MutableRefObject<any> = useRef(null);
  const outputEditor: React.MutableRefObject<any> = useRef(null);

  const [runArgsLeftAddon, setRunArgsLeftAddon] = useState("");
  const [runArgsRightAddon, setRunArgsRightAddon] = useState("");
  const [additionalRunArgs, setAdditionalRunArgs] = useState("");
  const [inputEditorFileName, setInputEditorFileName] = useState("");
  const [outputEditorFileName, setOutputEditorFileName] = useState("");

  const [stages, setStages] = useState<Array<StageState>>([new StageState()]);
  const [currentStageIdx, setCurrentStageIdx] = useState(0);

  function currentState() {
    return stages[currentStageIdx];
  }

  function updateState(updater: (state: StageState) => StageState) {
    setStages((prevState) => {
      return prevState.map((state, idx) => {
        return idx == currentStageIdx ? updater(state) : state;
      });
    });
  }

  function getCurrentPresetSelection() {
    return currentState().preset;
  }

  // Update the preset selection of the current stage.
  function setPresetSelection(selection: string) {
    updateState((oldState) => {
      let newStage = new StageState(selection);
      // TODO: Update editor content after confirming with user.
      const canChangeEditorContent = true;

      const presetProps = getPreset(selection);
      cppEditor.current.updateOptions({
        readOnly: !presetProps.isCodeEditorEnabled(),
      });
      setRunArgsLeftAddon(presetProps.getRunArgsLeftAddon());
      setRunArgsRightAddon(presetProps.getRunArgsRightAddon());
      setAdditionalRunArgs(presetProps.getDefaultAdditionalRunArgs());
      setInputEditorFileName(presetProps.getInputFileName());
      setOutputEditorFileName(presetProps.getOutputFileName());
      if (canChangeEditorContent) {
        cppEditor.current.setValue(presetProps.getDefaultCodeFile());
      }
      if (currentStageIdx == 0) {
        inputEditor.current.setValue(presetProps.getDefaultInputFile());
      }

      return newStage;
    });
  }

  function setCurrentLogs(updater: (log: Array<string>) => Array<string>) {
    updateState((oldState) => {
      let newState = { ...oldState };
      newState.logs = updater(newState.logs);
      return newState;
    });
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
        updateCompilerEnvironmentReady();
      }
    };
  };

  const onPresetSelectionChange = (
    event: React.ChangeEvent<HTMLSelectElement>
  ) => {
    setPresetSelection(event.target.value);
  };

  const onRunButtonClick = () => {
    setRunStatus("Initializing...");
    updateCompilerEnvironmentReady().then((isCached) => {
      const preset = getPreset(getCurrentPresetSelection());
      if (!isCached && preset.isCodeEditorEnabled()) {
        // Requires local compiler environment to be downloaded first.
        setCompilerEnvironmentPopoverOpen(true);
        setRunStatus("");
        return;
      }

      let cpp_source = "";
      if (preset.isCodeEditorEnabled()) {
        cpp_source = cppEditor.current.getValue();
        setRunStatus("Compiling...");
      } else {
        setRunStatus("Running...");
      }

      const input_mlir = inputEditor.current.getValue();
      setCurrentLogs((currValue) => [...currValue, ""]);
      const printer = (text: string) => {
        setCurrentLogs((currValue) => [
          ...currValue.slice(0, -1),
          currValue[currValue.length - 1] + text + "\n",
        ]);
      };
      const statusListener = (status: RunStatus) => {
        setRunStatus(status.label);
        setRunProgress(status.percentage);
        updateCompilerEnvironmentReady();
      };

      preset
        .run(cpp_source, input_mlir, additionalRunArgs, printer, statusListener)
        .finally(() => {
          setRunStatus("");
          setRunProgress(0);
        })
        .then((output: string) => {
          outputEditor.current.setValue(output);
        }, printer);
    });
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
        envVersion={compilerEnvironmentVersion}
        envPopoverOpen={compilerEnvironmentPopoverOpen}
        setEnvPopoverOpen={setCompilerEnvironmentPopoverOpen}
        initiateEnvDownload={downloadCompilerEnvironment}
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
        <VStack spacing={0}>
          <Button
            bg="none"
            rounded="none"
            width="100%"
            borderRightColor="gray.200"
            borderRightWidth="1px"
          >
            0
          </Button>
          <Button
            bg="none"
            rounded="none"
            width="100%"
            borderRightColor="gray.200"
            borderRightWidth="1px"
          >
            1
          </Button>
          <Button
            bg="none"
            rounded="none"
            width="100%"
            borderRightColor="gray.200"
            borderRightWidth="1px"
          >
            +
          </Button>
        </VStack>
        <Divider orientation="vertical" />
        <Box height="100%" className={styles.main_left}>
          <VStack spacing={4} align="left" height="100%">
            <PresetSelector
              preset={getCurrentPresetSelection()}
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
            logWindowProps={{ height: "30vh", logs: currentState().logs }}
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
