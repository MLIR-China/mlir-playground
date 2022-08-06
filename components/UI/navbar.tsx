import React from "react";
import { Flex, Heading, HStack, Image } from "@chakra-ui/react";

const NavBar = () => {
  return (
    <NavBarContainer>
      <Logo />
    </NavBarContainer>
  );
};

const Logo = () => {
  return (
    <HStack>
      <Image
        src="/mlir-playground.png"
        alt="MLIR Playground"
        boxSize="2em"
      />
      <Heading fontFamily="heading">MLIR Playground</Heading>
    </HStack>
  );
}

const NavBarContainer = ({ children }) => {
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
