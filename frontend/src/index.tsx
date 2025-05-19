
import React, { useState, useEffect, useRef } from "react";
import { debounce } from "lodash";
import Select from "react-select";
import {
  ChakraProvider,
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
  Heading,
  Flex,
  Box,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Input,
  Select as ChakraSelect,
  Button,
  extendTheme,
  SimpleGrid,
  VStack,
  HStack,
  Text,
  Avatar,
  Spacer
} from "@chakra-ui/react";
import { Bar } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
import type { ChartOptions } from "chart.js";

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

const theme = extendTheme({
  colors: {
    brand: {
      blue: "#003399",
      yellow: "#FFCC00"
    }
  }
});

const Header = () => (
  <Flex
    as="nav"
    align="center"
    justify="space-between"
    wrap="wrap"
    padding="1rem"
    bg="brand.blue"
    color="white"
    width="100%"
    position="fixed"
    top="0"
    left="0"
    right="0"
    zIndex="1000"
  >
    <Heading as="h1" size="md">EU Project Explorer</Heading>
  </Flex>
);

interface Project {
  startDate: string;
  id: string;
  title: string;
  status: string;
  ecMaxContribution: number;
}

function App() {
  const [question, setQuestion] = useState("");
  const [chatHistory, setChatHistory] = useState<{ role: string, content: string }[]>([]);
  const [projects, setProjects] = useState<Project[]>([]);
  const [stats, setStats] = useState<{ [key: string]: any }>({});
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(0);
  const [statusFilter, setStatusFilter] = useState("");
  const [selectedProject, setSelectedProject] = useState<Project | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [dashboardStatusFilter, setDashboardStatusFilter] = useState("");
  const [dashboardOrgFilter, setDashboardOrgFilter] = useState("");
  const [dashboardCountryFilter, setDashboardCountryFilter] = useState("");
  const [dashboardLegalBasisFilter, setDashboardLegalBasisFilter] = useState("");
  const [availableStatuses, setAvailableStatuses] = useState(["SIGNED", "CLOSED", "TERMINATED"]);
  const [availableOrgs, setAvailableOrgs] = useState<string[]>([]);
  const [availableCountries, setAvailableCountries] = useState<string[]>([]);
  const [availableLegalBases, setAvailableLegalBases] = useState<string[]>([]);

  useEffect(() => {
    const fetchFilters = (status: string, org: string, country: string, legalBasis: string) => {
      const params = new URLSearchParams({
        status,
        organization: org,
        country,
        legalBasis
      });
      fetch(`/api/filters?${params.toString()}`)
        .then(res => res.json())
        .then(data => {
          setAvailableOrgs(data.organizations);
          setAvailableCountries(data.countries);
          setAvailableLegalBases(data.legalBases);
        });
    };

    fetchFilters(dashboardStatusFilter, dashboardOrgFilter, dashboardCountryFilter, dashboardLegalBasisFilter);
  }, [dashboardStatusFilter, dashboardOrgFilter, dashboardCountryFilter, dashboardLegalBasisFilter]);

  const debouncedFetchStats = useRef(
    debounce((status, org, country, legalBasis) => {
      fetch(`/api/stats?status=${status}&organization=${org}&country=${country}&legalBasis=${legalBasis}`)
        .then(res => res.json())
        .then(setStats)
        .catch(console.error);
    }, 500)
  ).current;

  useEffect(() => {
    debouncedFetchStats(dashboardStatusFilter, dashboardOrgFilter, dashboardCountryFilter, dashboardLegalBasisFilter);
  }, [dashboardStatusFilter, dashboardOrgFilter, dashboardCountryFilter, dashboardLegalBasisFilter]);

  useEffect(() => {
    fetch(`/api/projects?page=${page}&search=${encodeURIComponent(search)}&status=${statusFilter}`)
      .then(res => {
        if (!res.ok) throw new Error("Failed to fetch projects");
        return res.json();
      })
      .then(data => setProjects(data))
      .catch(err => console.error("Error fetching projects:", err));
  }, [page, search, statusFilter]);

  useEffect(() => {
  fetch(`/api/stats?status=${dashboardStatusFilter}&organization=${dashboardOrgFilter}&country=${dashboardCountryFilter}&legalBasis=${dashboardLegalBasisFilter}`)
    .then(res => res.json())
    .then(setStats)
    .catch(console.error);
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatHistory]);

  const askChatbot = async () => {
      if (!question.trim()) return;
      const newChat = [...chatHistory, { role: "user", content: question }];
      setChatHistory(newChat);
      setQuestion("");
      try {
        const res = await fetch("/api/chat/query", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question })
        });
        const data = await res.json();
        setChatHistory([...newChat, { role: "assistant", content: data.answer }]);
      } catch {
        setChatHistory([...newChat, { role: "assistant", content: "Something went wrong." }]);
      }
    };

  const buildChart = (label: string, labels: string[], values: number[]) => ({
    data: {
      labels,
      datasets: [
        {
          label,
          data: values,
          backgroundColor: "#FFCC00",
          borderColor: "#003399",
          borderWidth: 1,
        }
      ]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { position: "top" },
        title: { display: true, text: label },
      },
    } as ChartOptions<'bar'>
  });

  return (
    <ChakraProvider theme={theme}>
      <Header />
      <Box paddingTop="80px" paddingX="4">
        <Tabs variant="enclosed-colored" colorScheme="yellow">
          <TabList>
            <Tab>Dashboard</Tab>
            <Tab>Projects + Chat</Tab>
            {selectedProject && <Tab>Project Details</Tab>}
          </TabList>

          <TabPanels>
            <TabPanel>
              <Heading size="md" mb={4}>Funding Overview</Heading>
              <Flex gap={4} mb={2} wrap="wrap">
                <Box w="200px">
                  <Select
                    options={availableStatuses.map(s => ({ label: s, value: s }))}
                    placeholder="Status"
                    onChange={(e) => setDashboardStatusFilter(e?.value || "")}
                    isClearable
                  />
                </Box>
                <Box w="200px">
                  <Select
                    options={availableOrgs.slice(0, 1000).map(o => ({ label: o, value: o }))}
                    placeholder="Organization"
                    onChange={(e) => setDashboardOrgFilter(e?.value || "")}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") {
                        const params = new URLSearchParams({
                          status: dashboardStatusFilter,
                          organization: dashboardOrgFilter,
                          country: dashboardCountryFilter,
                          legalBasis: dashboardLegalBasisFilter
                        });
                        fetch(`/api/filters?${params.toString()}`)
                          .then(res => res.json())
                          .then(data => {
                            setAvailableOrgs(data.organizations);
                            setAvailableCountries(data.countries);
                            setAvailableLegalBases(data.legalBases);
                          });
                      }
                    }}
                    isClearable/>
                </Box>
                <Box w="200px">
                    <Select
                    options={availableCountries.slice(0, 100).map(o => ({ label: o, value: o }))}
                    placeholder="Country"
                    onChange={(e) => setDashboardCountryFilter(e?.value || "")}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") {
                        const params = new URLSearchParams({
                          status: dashboardStatusFilter,
                          organization: dashboardOrgFilter,
                          country: dashboardCountryFilter,
                          legalBasis: dashboardLegalBasisFilter
                        });
                        fetch(`/api/filters?${params.toString()}`)
                          .then(res => res.json())
                          .then(data => {
                            setAvailableOrgs(data.organizations);
                            setAvailableCountries(data.countries);
                            setAvailableLegalBases(data.legalBases);
                          });
                      }
                    }}
                    isClearable/>
                </Box>
                <Box w="200px">
                    <Select
                    options={availableLegalBases.slice(0, 100).map(o => ({ label: o, value: o }))}
                    placeholder="Legal Basis"
                    onChange={(e) => setDashboardLegalBasisFilter(e?.value || "")}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") {
                        const params = new URLSearchParams({
                          status: dashboardStatusFilter,
                          organization: dashboardOrgFilter,
                          country: dashboardCountryFilter,
                          legalBasis: dashboardLegalBasisFilter
                        });
                        fetch(`/api/filters?${params.toString()}`)
                          .then(res => res.json())
                          .then(data => {
                            setAvailableOrgs(data.organizations);
                            setAvailableCountries(data.countries);
                            setAvailableLegalBases(data.legalBases);
                          });
                      }
                    }}
                    isClearable/>
                </Box>
              </Flex>
              <Flex gap={2} mb={4} wrap="wrap">
                {dashboardStatusFilter && <Box bg="yellow.200" px={2} py={1} borderRadius="md">Status: {dashboardStatusFilter}</Box>}
                {dashboardOrgFilter && <Box bg="yellow.200" px={2} py={1} borderRadius="md">Org: {dashboardOrgFilter}</Box>}
                {dashboardCountryFilter && <Box bg="yellow.200" px={2} py={1} borderRadius="md">Country: {dashboardCountryFilter}</Box>}
                {dashboardLegalBasisFilter && <Box bg="yellow.200" px={2} py={1} borderRadius="md">Legal Basis: {dashboardLegalBasisFilter}</Box>}
              </Flex>

              <SimpleGrid columns={{ base: 1, md: 2, lg: 3 }} spacing={6}>
                {Object.keys(stats).map((key, i) => {
                  const chart = buildChart(key, stats[key].labels, stats[key].values);
                  return (
                    <Box key={i} bg="white" borderRadius="md" p={4}>
                      <Bar data={chart.data} options={chart.options} />
                    </Box>
                  );
                })}
              </SimpleGrid>
            </TabPanel>

            <TabPanel>
              <Flex direction={{ base: "column", md: "row" }} gap={6}>
                <Box flex={1}>
                  <Heading size="sm" mb={2}>Projects</Heading>
                  <Flex gap={4} mb={4}>
                    <Input
                      placeholder="Search by title..."
                      value={search}
                      onChange={(e) => { setSearch(e.target.value); setPage(0); }}
                    />
                    <ChakraSelect
                      placeholder="Filter by status"
                      value={statusFilter}
                      onChange={(e) => { setStatusFilter(e.target.value); setPage(0); }}
                    >
                      <option value="Signed">Signed</option>
                      <option value="Closed">Closed</option>
                      <option value="Terminated">Terminated</option>
                    </ChakraSelect>
                  </Flex>
                  <Table variant="simple" size="sm">
                    <Thead>
                      <Tr>
                        <Th>Title</Th>
                        <Th>Status</Th>
                        <Th>ID</Th>
                        <Th>Start Date</Th>
                        <Th>Funding €</Th>
                      </Tr>
                    </Thead>
                    <Tbody>
                      {projects.map((p, i) => (
                        <Tr key={i} onClick={() => setSelectedProject(p)} cursor="pointer">
                          <Td>{p.title}</Td>
                          <Td>{p.status}</Td>
                          <Td>{p.id}</Td>
                          <Td>{p.startDate}</Td>
                          <Td>{p.ecMaxContribution?.toLocaleString()}</Td>
                        </Tr>
                      ))}
                    </Tbody>
                  </Table>
                  <Flex mt={4} gap={2}>
                    <Button onClick={() => setPage(p => Math.max(p - 1, 0))} isDisabled={page === 0}>Previous</Button>
                    <Button onClick={() => setPage(p => p + 1)}>Next</Button>
                  </Flex>
                </Box>
                <Box flex={0.6} display="flex" flexDirection="column" bg="gray.50" p={4} borderRadius="md" height="500px">
                  <Heading size="sm" mb={2}>Assistant</Heading>
                  <Box flex={1} overflowY="auto" mb={4}>
                    <VStack spacing={3} align="stretch">
                      {chatHistory.map((msg, i) => (
                        <HStack key={i} alignSelf={msg.role === "user" ? "flex-end" : "flex-start"} maxW="90%">
                          {msg.role === "assistant" && <Avatar size="sm" name="Bot" />}
                          <Box>
                            <Text fontSize="sm" bg={msg.role === "user" ? "blue.100" : "gray.200"} px={3} py={2} borderRadius="md">
                              {msg.content}
                            </Text>
                          </Box>
                          {msg.role === "user" && <Avatar size="sm" name="You" bg="blue.300" />}
                        </HStack>
                      ))}
                      <div ref={messagesEndRef} />
                    </VStack>
                  </Box>
                  <HStack>
                    <Input
                      placeholder="Ask something..."
                      value={question}
                      onChange={e => setQuestion(e.target.value)}
                      onKeyDown={e => { if (e.key === 'Enter') { e.preventDefault(); askChatbot(); } }}
                    />
                    <Button onClick={askChatbot} colorScheme="blue">Send</Button>
                  </HStack>
                </Box>
              </Flex>
            </TabPanel>

            {selectedProject && (
              <TabPanel>
                <Heading size="md" mb={4}>{selectedProject.title}</Heading>
                <p><strong>ID:</strong> {selectedProject.id}</p>
                <p><strong>Status:</strong> {selectedProject.status}</p>
                <p><strong>Start Date:</strong> {selectedProject.startDate}</p>
                <p><strong>Funding:</strong> €{selectedProject.ecMaxContribution.toLocaleString()}</p>
              </TabPanel>
            )}
          </TabPanels>
        </Tabs>
      </Box>
    </ChakraProvider>
  );
}

export default App;
