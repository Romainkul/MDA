import React, { useState, useEffect } from "react";
import {
  ChakraProvider,
  extendTheme,
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
  Container,
  Box,
  Text,
} from "@chakra-ui/react";
import Header from "./components/Header";
import Dashboard from "./components/Dashboard";
import ProjectExplorer from "./components/ProjectExplorer";
import ProjectDetails from "./components/ProjectDetails";
import { useAppState } from "./hooks/useAppState";

const theme = extendTheme({
  fonts: {
    heading: `'Segoe UI', sans-serif`,
    body: `'Segoe UI', sans-serif`,
  },
  colors: {
    brand: {
      blue: "#003399", 
      yellow: "#FFCC00", 
      gray: "#f5f5f5",
    },
  },
  styles: {
    global: {
      html: {
      bg: "white", // ✅ add this
    },
      body: {
        bg: "white",
        color: "gray.800",
      },
    },
  },
  components: {
    Tabs: {
      baseStyle: {
        tab: {
          _selected: {
            bg: "brand.yellow",
            color: "brand.blue",
            fontWeight: "bold",
            borderColor: "brand.blue",
            borderTopWidth: "4px",
          },
        },
        tabpanel: {
          borderTopWidth: "0px",
        },
      },
    },
    Button: {
      variants: {
        solid: {
          bg: "brand.blue",
          color: "white",
          _hover: { bg: "#002080" },
        },
      },
    },
  },
  radii: {
    md: "12px",
    lg: "16px",
    xl: "24px",
  },
});

export default function App() {
  const [tabIndex, setTabIndex] = useState(0);
  const {
    selectedProject,
    dashboardProps,
    explorerProps,
    detailsProps,
  } = useAppState();

  // If they deselect, never stay on “Details”
  useEffect(() => {
    if (!selectedProject && tabIndex === 2) {
      setTabIndex(1);
    }
  }, [selectedProject, tabIndex]);

  // If they just selected a project, jump to details
  useEffect(() => {
    if (selectedProject) {
      setTabIndex(2);
    }
  }, [selectedProject]);

  return (
    <ChakraProvider theme={theme}>
      <Header />
      <Container maxW="9xl" mt="90px" px={{ base: 4, md: 8 }}>
        <Box bg="white" borderRadius="xl" boxShadow="lg" p={6}>
          <Tabs
            variant="enclosed-colored"
            colorScheme="yellow"
            index={tabIndex}
            onChange={setTabIndex}
            isFitted
            aria-label="Main Navigation Tabs"
          >
            <TabList role="tablist">
              <Tab>Dashboard</Tab>
              <Tab>Projects</Tab>
              <Tab isDisabled={!selectedProject}>Project Details</Tab>
            </TabList>

            <TabPanels>
              <TabPanel>
                <Dashboard {...dashboardProps} />
              </TabPanel>
              <TabPanel>
                <ProjectExplorer {...explorerProps} />
              </TabPanel>
              <TabPanel>
                {selectedProject
                  ? <ProjectDetails {...detailsProps} />
                  : <Text color="gray.500" textAlign="center" py={10}>
                      Select a project to see its details
                    </Text>
                }
              </TabPanel>
            </TabPanels>
          </Tabs>
        </Box>
      </Container>
    </ChakraProvider>
  );
}