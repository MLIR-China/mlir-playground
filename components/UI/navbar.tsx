import React from "react";
import {
  BoxProps,
  Button,
  Flex,
  Heading,
  HStack,
  Icon,
  Image,
  LinkBox,
  LinkOverlay,
  Popover,
  PopoverArrow,
  PopoverBody,
  PopoverCloseButton,
  PopoverContent,
  PopoverFooter,
  PopoverHeader,
  PopoverTrigger,
} from "@chakra-ui/react";
import { GoMarkGithub } from "react-icons/go";
import { MdOutlineCode, MdOutlineCodeOff } from "react-icons/md";

type NavBarProps = LocalEnvironmentStatusProps;

const NavBar = (props: NavBarProps) => {
  return (
    <NavBarContainer>
      <HStack spacing="1rem">
        <Logo />
        <LocalEnvironmentStatus {...props} />
      </HStack>
      <RightEndButtons />
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

type LocalEnvironmentStatusProps = {
  envVersion: string; // empty string means not ready
  envPopoverOpen: boolean;
  setEnvPopoverOpen: (isOpen: boolean) => void;
  initiateEnvDownload: () => Promise<boolean>;
};

const LocalEnvironmentStatus = (props: LocalEnvironmentStatusProps) => {
  const popoverMessage = `MLIR Playground compiles and runs all your code locally in your browser.
This requires downloading the necessary binaries and libraries (including libc++ and MLIR libraries), which will be cached for future sessions (until evicted by the browser or the user).
This will incur a download of ~100MB once.`;
  const downloadButtonRef = React.useRef<HTMLButtonElement>(null);
  const [downloadInProgress, setDownloadInProgress] = React.useState(false);
  const onDownloadButtonClick = () => {
    setDownloadInProgress(true);
    props.initiateEnvDownload().finally(() => {
      setDownloadInProgress(false);
    });
  };

  const envReady = !!props.envVersion;

  return (
    <Popover
      returnFocusOnClose={false}
      closeOnBlur={false}
      initialFocusRef={envReady ? undefined : downloadButtonRef}
      isOpen={props.envPopoverOpen}
      onClose={() => props.setEnvPopoverOpen(false)}
    >
      <PopoverTrigger>
        <Button
          leftIcon={envReady ? <MdOutlineCode /> : <MdOutlineCodeOff />}
          borderRadius="full"
          backgroundColor={envReady ? "green.200" : "gray.200"}
          onClick={() => props.setEnvPopoverOpen(true)}
        >
          {envReady ? "Ready" : "Pending"}
        </Button>
      </PopoverTrigger>
      <PopoverContent>
        <PopoverHeader fontWeight="semibold">
          Local compiler environment
        </PopoverHeader>
        <PopoverArrow />
        <PopoverCloseButton />
        <PopoverBody style={{ whiteSpace: "pre-wrap" }}>
          <p>{popoverMessage}</p>
          {envReady && <i>Local Version: {props.envVersion}</i>}
        </PopoverBody>
        <PopoverFooter d="flex" justifyContent="flex-end">
          <Button
            ref={downloadButtonRef}
            isDisabled={envReady}
            isLoading={downloadInProgress}
            onClick={onDownloadButtonClick}
          >
            {envReady ? "Downloaded" : "Download"}
          </Button>
        </PopoverFooter>
      </PopoverContent>
    </Popover>
  );
};

const RightEndButtons = (props: BoxProps) => {
  return (
    <LinkBox title="View source code on GitHub" {...props}>
      <Button bg="none" rightIcon={<Icon as={GoMarkGithub} />}>
        <LinkOverlay href="https://github.com/MLIR-China/mlir-playground">
          GitHub
        </LinkOverlay>
      </Button>
    </LinkBox>
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
