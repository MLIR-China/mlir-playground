import React, { Fragment, useEffect, useRef, useState } from "react";
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
  Progress,
  Select,
  Tab,
  Tabs,
  TabList,
  Text,
  Tooltip,
  useToast,
  UseToastOptions,
  VStack,
  ButtonGroup,
} from "@chakra-ui/react";
import { OnMount } from "@monaco-editor/react";
import { usePlausible } from "next-plausible";

import styles from "../styles/Home.module.css";
import { monospaceFontFamily } from "../components/UI/constants";

import {
  defaultPreset,
  getPresetNames,
  getPreset,
  presetOption,
} from "../components/Presets/PresetFactory";

import LabeledEditor from "../components/UI/labeledEditor";
import NavBar from "../components/UI/navbar";
import WasmCompiler from "../components/WasmCompiler";
import { AllPlaygroundEvents } from "../components/Utils/Events";
import { RunStatus } from "../components/Utils/RunStatus";
import {
  PlaygroundPreset,
  PlaygroundPresetPane,
} from "../components/Presets/PlaygroundPreset";
import {
  exportToSchema,
  importFromSchema,
  SchemaObjectType,
} from "../components/State/ImportExport";
import {
  StageState,
  newStageStateFromPreset,
  stageStateIsDirty,
} from "../components/State/StageState";

import { validateAgainstSchema } from "../schema/validation";

function getInputFileBaseName(stageIndex: number) {
  const prevIndex = stageIndex - 1;
  return stageIndex == 0 ? "input" : `output${prevIndex}`;
}

function getOutputFileBaseName(stageIndex: number) {
  return `output${stageIndex}`;
}

const Home: NextPage = () => {
  const logEvent = usePlausible<AllPlaygroundEvents>();
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

    logEvent("EnvDownloadStart");
    return WasmCompiler.initialize().then((success) => {
      logEvent("EnvDownloadDone", { props: { success: success } });
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

  // Stores the entire state across all stages.
  // _rawSetCurrentStageIdx should never be used directly (always use the wrapper setCurrentStageIdx).
  const [stages, setStages] = useState<Array<StageState>>([
    newStageStateFromPreset(),
  ]);
  const [currentStageIdx, _rawSetCurrentStageIdx] = useState(0);
  const [inputEditorContent, setInputEditorContent] = useState("");

  function currentStage() {
    return stages[currentStageIdx];
  }

  function getCurrentPresetSelection() {
    return currentStage().preset;
  }

  function updateCurrentStage(
    partialStateGetter:
      | ((oldState: StageState) => Partial<StageState>)
      | Partial<StageState>
  ) {
    setStages((prevStages) => {
      return prevStages.map((state, idx) => {
        if (idx != currentStageIdx) {
          return state;
        }
        const partialState: Partial<StageState> =
          typeof partialStateGetter == "function"
            ? partialStateGetter(state)
            : partialStateGetter;
        return { ...state, ...partialState };
      });
    });
  }

  function appendStage() {
    setStages((prevStages) => {
      return [...prevStages, newStageStateFromPreset()];
    });
  }

  function getOutputFileName(stageIndex: number) {
    const presetProps = getPreset(stages[stageIndex].preset);
    return (
      getOutputFileBaseName(stageIndex) +
      "." +
      presetProps.getOutputFileExtension()
    );
  }

  function updateAuxiliaryInformation(
    currentIndex: number,
    presetProps: PlaygroundPreset
  ) {
    const inputFileName =
      getInputFileBaseName(currentIndex) +
      "." +
      presetProps.getInputFileExtension();
    const outputFileName =
      getOutputFileBaseName(currentIndex) +
      "." +
      presetProps.getOutputFileExtension();
    setRunArgsLeftAddon(
      presetProps.getRunArgsLeftAddon(inputFileName, outputFileName)
    );
    setRunArgsRightAddon(
      presetProps.getRunArgsRightAddon(inputFileName, outputFileName)
    );
    setInputEditorFileName(inputFileName);
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

    // update raw state
    _rawSetCurrentStageIdx(idx);

    // additional side effects
    const newStage = stages[idx];
    const presetProps = getPreset(newStage.preset);
    updateAuxiliaryInformation(idx, presetProps);

    newStage.outputEditorWindow!.current.scrollIntoView({
      behavior: "smooth",
      block: "end",
      inline: "nearest",
    });
  }

  // Returns true if any editor value is non-empty and not the same as the default for their selected preset.
  // If currentOnly, then only the current editor is checked.
  function isEditorDirty(currentOnly: boolean) {
    if (currentOnly) {
      return stageStateIsDirty(currentStage());
    }
    return stages.some((stage) => {
      return stageStateIsDirty(stage);
    });
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
  function setPresetSelection(selection: presetOption) {
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

    const presetProps = getPreset(selection);

    if (currentStageIdx > 0 && !presetProps.isMultiStageCompatible()) {
      toast({
        title: `Cannot use "${selection}" in multi-stage mode.`,
        description: `"${selection}" is only available in stage 0. Please switch to stage 0 first.`,
        status: "warning",
        position: "top",
        isClosable: true,
        duration: null,
      });
      return;
    }

    updateCurrentStage({
      preset: selection,
      additionalRunArgs: presetProps.getDefaultAdditionalRunArgs(),
      editorContents: presetProps
        .getPanes()
        .map((pane: PlaygroundPresetPane) => {
          return pane.defaultEditorContent;
        }),
    });
    updateAuxiliaryInformation(currentStageIdx, presetProps);
    if (currentStageIdx == 0) {
      setInputEditorContent(presetProps.getDefaultInputFile());
    }
  }

  function setCurrentLogs(updater: (log: Array<string>) => Array<string>) {
    updateCurrentStage((oldState) => {
      return { logs: updater(oldState.logs) };
    });
  }

  const onAllEditorsMounted = () => {
    // Import from url params.
    const urlParams = new URLSearchParams(window.location.search);
    let importUrl = urlParams.get("import");
    if (importUrl) {
      // Allow user to omit "http(s)://" prefix.
      if (!/^https?:\/\//i.test(importUrl)) {
        importUrl = "https://" + importUrl;
      }

      fetch(importUrl)
        .then((response) => {
          if (response.ok) {
            return response.json();
          }
          return Promise.reject("Failed to fetch resource.");
        })
        .then((parsedObject) => {
          let errorMsg = importFromSchemaObject(parsedObject);
          if (!errorMsg) {
            toast({
              title: "Import Success!",
              description: "Successfully imported from URL.",
              status: "success",
              position: "top",
            });
            return;
          }
          return Promise.reject("Failed to parse file: " + errorMsg);
        })
        .catch((error) => {
          const errorTitle = `Failed to import from URL: ${importUrl}`;
          console.log(errorTitle + "\n" + error);
          toast({
            title: errorTitle,
            description: String(error),
            status: "error",
            position: "top",
            isClosable: true,
            duration: null,
          });
        });
    }

    // This will unset the loading state of buttons.
    setAllEditorsMounted(true);
    setPresetSelection(defaultPreset);
    updateCompilerEnvironmentReady();
  };

  const onEditorMounted = (editorRef: React.MutableRefObject<any>): OnMount => {
    return (editor, _) => {
      editorRef.current = editor;
      window.addEventListener("resize", () => {
        editorRef.current.layout({});
      });

      if (
        cppEditor.current &&
        inputEditor.current &&
        currentStage().outputEditor.current &&
        !allEditorsMounted
      ) {
        onAllEditorsMounted();
      }
    };
  };

  const onPresetSelectionChange = (
    event: React.ChangeEvent<HTMLSelectElement>
  ) => {
    // This is considered implicitly safe since the options are provided by us.
    setPresetSelection(event.target.value as presetOption);
  };

  // Dims the old logs and returns a function for adding log messages.
  const getNextLogger = () => {
    setCurrentLogs((currValue) => [...currValue, ""]);
    return (text: string) => {
      setCurrentLogs((currValue) => [
        ...currValue.slice(0, -1),
        currValue[currValue.length - 1] + text + "\n",
      ]);
    };
  };

  const onActionButtonClick = (actionName: string) => {
    setRunStatus("Performing Action...");
    updateCompilerEnvironmentReady().then((isCached) => {
      if (!isCached) {
        setCompilerEnvironmentPopoverOpen(true);
        setRunStatus("");
        return;
      }

      const editorContents = currentStage().editorContents;
      const printer = getNextLogger();
      const preset = getPreset(currentStage().preset);
      preset
        .getActions()
        [actionName](editorContents, printer)
        .finally(() => {
          setRunStatus("");
        })
        .then((outputs) => {
          updateCurrentStage({ editorContents: outputs });
          printer("Completed: " + actionName);
        });
    });
  };

  const onRunButtonClick = () => {
    setRunStatus("Initializing...");
    updateCompilerEnvironmentReady().then((isCached) => {
      const preset = getPreset(getCurrentPresetSelection());
      if (!isCached && preset.getPanes().length > 0) {
        // Has at least one editor: Requires local compiler environment to be downloaded first.
        setCompilerEnvironmentPopoverOpen(true);
        setRunStatus("");
        return;
      }

      const editorContents = currentStage().editorContents;
      if (editorContents.length > 0) {
        setRunStatus("Compiling...");
      } else {
        setRunStatus("Running...");
      }

      const input_mlir =
        currentStageIdx == 0
          ? inputEditorContent
          : stages[currentStageIdx - 1].output;

      const printer = getNextLogger();
      const statusListener = (status: RunStatus) => {
        setRunStatus(status.label);
        setRunProgress(status.percentage);
        updateCompilerEnvironmentReady();
      };

      logEvent("RunStart", { props: { preset: getCurrentPresetSelection() } });
      preset
        .run(
          editorContents,
          input_mlir,
          currentStage().additionalRunArgs,
          printer,
          statusListener
        )
        .finally(() => {
          setRunStatus("");
          setRunProgress(0);
          logEvent("RunEnd");
        })
        .then((output: string) => {
          updateCurrentStage({ output: output });
          printer("Completed: Run");
        }, printer);
    });
  };

  const exportToSchemaObject = () => {
    return exportToSchema(
      { input: inputEditorContent, stages: stages },
      compilerEnvironmentVersion
    );
  };

  // Returns an error message if failed. Otherwise returns an empty string.
  const importFromSchemaObject = (source: any) => {
    const validationError = validateAgainstSchema(source);
    if (validationError) {
      return validationError;
    }

    const internalState = importFromSchema(source as SchemaObjectType);
    if (typeof internalState === "string") {
      // Error during import
      return internalState;
    }

    setInputEditorContent(internalState.input);
    setStages(internalState.stages);
    return "";
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
        envVersion={compilerEnvironmentVersion}
        envPopoverOpen={compilerEnvironmentPopoverOpen}
        setEnvPopoverOpen={setCompilerEnvironmentPopoverOpen}
        initiateEnvDownload={downloadCompilerEnvironment}
        exportToSchemaObject={exportToSchemaObject}
        importFromSchemaObject={importFromSchemaObject}
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
        <Flex direction="column">
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
            title="Append a new stage"
          >
            +
          </Button>
        </Flex>
        <Divider orientation="vertical" />
        <Flex
          flexDirection="column"
          align="left"
          className={styles.main_left}
          height="100%"
          width="100%"
        >
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
              updateCurrentStage({ additionalRunArgs: newArgs });
            }}
          />
          <Flex flexDirection="column" width="100%" flex="1">
            {currentStage().editorContents.length > 0 ? (
              <Fragment>
                <Tabs
                  index={currentStage().currentPaneIdx!}
                  onChange={(newIndex) => {
                    updateCurrentStage({ currentPaneIdx: newIndex });
                  }}
                >
                  <TabList>
                    {getPreset(currentStage().preset)
                      .getPanes()
                      .map((pane, index) => {
                        return <Tab key={index}>{pane.shortName}</Tab>;
                      })}
                  </TabList>
                </Tabs>
                <LabeledEditor
                  flex="1"
                  filename={`editor-${currentStageIdx}-${currentStage()
                    .currentPaneIdx!}`}
                  onMount={onEditorMounted(cppEditor)}
                  value={
                    currentStage().editorContents[
                      currentStage().currentPaneIdx!
                    ]
                  }
                  onChange={(value, event) => {
                    if (value) {
                      updateCurrentStage((prevState) => {
                        return {
                          editorContents: prevState.editorContents.map(
                            (prevValue, index) => {
                              return index == prevState.currentPaneIdx
                                ? value
                                : prevValue;
                            }
                          ),
                        };
                      });
                    }
                  }}
                />
              </Fragment>
            ) : (
              <Text mt="0.5rem">
                Editor is not needed for the current Preset.
              </Text>
            )}
          </Flex>
          <Flex justifyContent="space-between" marginTop={2} width="100%">
            <ButtonGroup isAttached>
              {Object.keys(getPreset(currentStage().preset).getActions()).map(
                (actionName) => {
                  return (
                    <Button
                      key={actionName}
                      onClick={() => onActionButtonClick(actionName)}
                      isLoading={!allEditorsMounted || runStatus !== ""}
                    >
                      {actionName}
                    </Button>
                  );
                }
              )}
            </ButtonGroup>
            <RunButton
              allEditorsMounted={allEditorsMounted}
              runStatus={runStatus}
              runProgress={runProgress}
              onClick={onRunButtonClick}
            />
          </Flex>
        </Flex>
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
            value={inputEditorContent}
            onChange={(value, event) => {
              if (value) {
                setInputEditorContent(value);
              }
            }}
          />
          {stages.map((stage, idx) => (
            <TransformationOutput
              logWindowProps={{ height: "30vh", logs: stage.logs }}
              labeledEditorProps={{
                height: "30vh",
                label: `Output ${idx}`,
                filename: getOutputFileName(idx),
                onMount: onEditorMounted(stage.outputEditor),
                ref: stage.outputEditorWindow,
                value: stage.output,
                onChange: (value, event) => {
                  if (value) {
                    updateCurrentStage({ output: value });
                  }
                },
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
    <HStack className={styles.preset_selector_bar}>
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
    <HStack className={styles.arguments_bar}>
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

type RunButtonProps = {
  allEditorsMounted: boolean;
  runStatus: string;
  runProgress: number;
  onClick: () => void;
};

const RunButton = (props: RunButtonProps) => {
  return (
    <HStack h="100%">
      {props.runProgress > 0 && (
        <Tooltip hasArrow label={props.runStatus}>
          <span>
            <Progress
              value={props.runProgress}
              w="10rem"
              hasStripe
              isAnimated
              borderRadius="md"
              mr="0.5rem"
              sx={{
                "& > div:first-of-type": {
                  transitionProperty: "width",
                },
              }}
            />
          </span>
        </Tooltip>
      )}
      <Button
        isLoading={!props.allEditorsMounted || props.runStatus !== ""}
        as="a"
        colorScheme="blue"
        fontWeight="bold"
        onClick={props.onClick}
      >
        Run
      </Button>
    </HStack>
  );
};

export default Home;
