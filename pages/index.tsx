import React, { useEffect, useRef, useState } from "react";
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
  Text,
  useToast,
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
import { PlaygroundPreset } from "../components/Presets/PlaygroundPreset";
import { MdOutlineSdStorage } from "react-icons/md";

// Stores the configuration of a particular stage.
class StageState {
  preset: string;
  additionalRunArgs: string;
  editorContent: string;
  logs: Array<string>;
  output: string;

  outputEditor: React.MutableRefObject<any> = React.createRef();
  outputEditorWindow: React.MutableRefObject<any> = React.createRef();

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
  const toast = useToast();

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

  const [runArgsLeftAddon, setRunArgsLeftAddon] = useState("");
  const [runArgsRightAddon, setRunArgsRightAddon] = useState("");
  const [inputEditorFileName, setInputEditorFileName] = useState("");
  const [outputEditorFileName, setOutputEditorFileName] = useState("");

  // Stores the entire state across all stages.
  // _rawSetCurrentStageIdx should never be used directly (always use the wrapper setCurrentStageIdx).
  const [stages, setStages] = useState<Array<StageState>>([new StageState()]);
  const [currentStageIdx, _rawSetCurrentStageIdx] = useState(0);

  function currentStage() {
    return stages[currentStageIdx];
  }

  function updateState(updater: (state: StageState) => StageState) {
    setStages((prevStages) => {
      return prevStages.map((state, idx) => {
        return idx == currentStageIdx ? updater(state) : state;
      });
    });
  }

  function appendStage() {
    setStages((prevStages) => {
      return [...prevStages, new StageState()];
    });
  }

  function updateAuxiliaryInformation(presetProps: PlaygroundPreset) {
    cppEditor.current.updateOptions({
      readOnly: !presetProps.isCodeEditorEnabled(),
    });
    setRunArgsLeftAddon(presetProps.getRunArgsLeftAddon());
    setRunArgsRightAddon(presetProps.getRunArgsRightAddon());
    setInputEditorFileName(presetProps.getInputFileName());
    setOutputEditorFileName(presetProps.getOutputFileName());
  }

  function setCurrentStageIdx(idx: number) {
    if (idx == currentStageIdx) {
      return;
    }

    // make sure nothing is running
    if (runStatus) {
      toast({
        title: "Cannot change stage.",
        description: "Job is currently running.",
        status: "warning",
        position: "top",
      });
      return;
    }

    // save current editor state
    updateState((oldState) => {
      let newState = { ...oldState };
      newState.editorContent = cppEditor.current.getValue();
      return newState;
    });

    // update raw state
    _rawSetCurrentStageIdx(idx);

    // additional side effects
    const newStage = stages[idx];
    const presetProps = getPreset(newStage.preset);
    updateAuxiliaryInformation(presetProps);
    cppEditor.current.setValue(newStage.editorContent);

    newStage.outputEditorWindow!.current.scrollIntoView({
      behavior: "smooth",
      block: "end",
      inline: "nearest",
    });
  }

  function getCurrentPresetSelection() {
    return currentStage().preset;
  }

  // Returns true if any editor value is non-empty and not the same as the default for their selected preset.
  // If currentOnly, then only the current editor is checked.
  function isEditorDirty(currentOnly: boolean) {
    let isDirty = false;
    const presetProps = getPreset(getCurrentPresetSelection());
    const currentEditorValue = cppEditor.current.getValue();
    isDirty ||=
      currentEditorValue.trim().length > 0 &&
      currentEditorValue != presetProps.getDefaultCodeFile();

    if (!currentOnly) {
      isDirty ||= stages.some((stage, index) => {
        const presetProps = getPreset(stage.preset);

        if (index == currentStageIdx) {
          return false; // Already checked above.
        }

        return (
          stage.editorContent.trim().length > 0 &&
          stage.editorContent != presetProps.getDefaultCodeFile()
        );
      });
    }

    return isDirty;
  }

  const warnUnsavedChanges = (event: BeforeUnloadEvent) => {
    if (isEditorDirty(false)) {
      event.preventDefault();
      return (event.returnValue = "Are you sure you want to exit?");
    }
  };

  useEffect(() => {
    window.addEventListener("beforeunload", warnUnsavedChanges);
    return () => {
      window.removeEventListener("beforeunload", warnUnsavedChanges);
    };
  });

  // Update the preset selection of the current stage.
  function setPresetSelection(selection: string) {
    if (isEditorDirty(true)) {
      // Check with the user first. If dialogs are disabled, this will always return false.
      if (
        !window.confirm(
          "Do you want the new preset to override your existing code?"
        )
      ) {
        return;
      }
    }

    updateState((oldState) => {
      let newStage = { ...oldState };
      const presetProps = getPreset(selection);
      newStage.preset = selection;
      newStage.additionalRunArgs = presetProps.getDefaultAdditionalRunArgs();
      updateAuxiliaryInformation(presetProps);
      cppEditor.current.setValue(presetProps.getDefaultCodeFile());
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
      window.addEventListener("resize", () => {
        editorRef.current.layout({});
      });

      if (
        cppEditor.current &&
        inputEditor.current &&
        currentStage().outputEditor.current
      ) {
        // All editors mounted.
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

      const input_mlir =
        currentStageIdx == 0
          ? inputEditor.current.getValue()
          : stages[currentStageIdx - 1].outputEditor.current.getValue();

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
        .run(
          cpp_source,
          input_mlir,
          currentStage().additionalRunArgs,
          printer,
          statusListener
        )
        .finally(() => {
          setRunStatus("");
          setRunProgress(0);
        })
        .then((output: string) => {
          currentStage().outputEditor.current.setValue(output);
        }, printer);
    });
  };

  return (
    <VStack className={styles.container}>
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
        width="100%"
        padding="0.5rem 1rem 0 1rem"
      >
        <VStack spacing={0}>
          {stages.map((_, idx) => {
            return (
              <Button
                bg="none"
                rounded="none"
                width="100%"
                borderRightColor={
                  idx == currentStageIdx ? "blue.200" : "gray.200"
                }
                borderRightWidth={idx == currentStageIdx ? "2px" : "1px"}
                key={idx}
                onClick={() => {
                  setCurrentStageIdx(idx);
                }}
              >
                {idx}
              </Button>
            );
          })}
          <Button
            bg="none"
            rounded="none"
            width="100%"
            borderRightColor="gray.200"
            borderRightWidth="1px"
            onClick={appendStage}
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
              additionalRunArgs={currentStage().additionalRunArgs}
              setAdditionalRunArgs={(newArgs) => {
                currentStage().additionalRunArgs = newArgs;
              }}
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
        <VStack
          height="100%"
          className={styles.main_right}
          overflow="hidden"
          spacing={0}
        >
          <LabeledEditor
            height="30vh"
            label="Input"
            filename={inputEditorFileName}
            onMount={onEditorMounted(inputEditor)}
          />
          {stages.map((stage, idx) => (
            <TransformationOutput
              logWindowProps={{ height: "100%", logs: stage.logs }}
              labeledEditorProps={{
                height: "30vh",
                label: `Output ${idx}`,
                filename: outputEditorFileName,
                onMount: onEditorMounted(stage.outputEditor),
                ref: stage.outputEditorWindow,
              }}
              key={idx}
            />
          ))}
        </VStack>
      </Flex>
    </VStack>
  );
};

type TransformationOutputProps = {
  logWindowProps: Parameters<typeof LogWindow>[0];
  labeledEditorProps: Parameters<typeof LabeledEditor>[0];
};

const TransformationOutput = (props: TransformationOutputProps) => {
  return (
    <Flex flexDirection="column" width="100%">
      <HStack flexGrow="1">
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
