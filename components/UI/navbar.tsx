import React from "react";
import {
  Box,
  Button,
  Flex,
  Heading,
  HStack,
  Icon,
  Image,
  Progress,
  Tag,
  TagLabel,
  Tooltip,
} from "@chakra-ui/react";
import { MdOutlineCode, MdOutlineCodeOff } from "react-icons/md";

type NavBarProps = RunButtonProps & { localEnvironmentReady: boolean };

const NavBar = (props: NavBarProps) => {
  return (
    <NavBarContainer>
      <HStack spacing="1rem">
        <Logo />
        <LocalEnvironmentStatus ready={props.localEnvironmentReady} />
      </HStack>
      <RunButton {...props} />
    </NavBarContainer>
  );
};

const Logo = () => {
  return (
    <HStack>
      <Image src="/mlir-playground.png" alt="MLIR Playground" boxSize="2em" />
      <Heading fontFamily="heading">MLIR Playground</Heading>
    </HStack>
  );
};

const LocalEnvironmentStatus = (props: { ready: boolean }) => {
  const readyMessage = "Compiler environment cached";
  const notReadyMessage =
    "Compiler environment will be cached on first compile.";
  return (
    <Tooltip hasArrow label={props.ready ? readyMessage : notReadyMessage}>
      <Tag
        size="lg"
        borderRadius="full"
        backgroundColor={props.ready ? "green.200" : "gray.200"}
        color="gray.700"
      >
        <Icon as={props.ready ? MdOutlineCode : MdOutlineCodeOff} mr="0.5rem" />
        <TagLabel>{props.ready ? "Ready" : "Pending"}</TagLabel>
      </Tag>
    </Tooltip>
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
        <Progress
          value={props.runProgress}
          w="10rem"
          hasStripe
          isAnimated
          borderRadius="md"
        />
      )}
      <Box>{props.runStatus}</Box>
      <Button
        isLoading={!props.allEditorsMounted || props.runStatus !== ""}
        as="a"
        size="lg"
        colorScheme="blue"
        fontWeight="bold"
        onClick={props.onClick}
      >
        Run
      </Button>
    </HStack>
  );
};

const NavBarContainer = ({ children }: { children: React.ReactNode }) => {
  return (
    <Flex
      as="nav"
      align="center"
      justify="space-between"
      w="100%"
      boxShadow="md"
      padding="0.5rem 1rem"
    >
      {children}
    </Flex>
  );
};

export default NavBar;
