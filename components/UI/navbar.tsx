import React from "react";
import {
  Box,
  Button,
  Flex,
  Heading,
  HStack,
  Image,
  Text,
} from "@chakra-ui/react";

const NavBar = (props: RunButtonProps) => {
  return (
    <NavBarContainer>
      <Logo />
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

type RunButtonProps = {
  allEditorsMounted: boolean;
  runStatus: string;
  onClick: () => void;
};

const RunButton = (props: RunButtonProps) => {
  return (
    <HStack h="100%">
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
