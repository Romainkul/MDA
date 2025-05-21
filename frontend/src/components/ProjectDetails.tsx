import { useEffect, useState } from "react";//React,
import {
  Box,
  Flex,
  Heading,
  Text,
  Spinner,
  SimpleGrid,
  Badge,
  Wrap,
  Tag,
  Divider,
  VStack,
  HStack,
  Input,
  Button,
  Avatar,
} from "@chakra-ui/react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip
} from "recharts";
import {
  MapContainer,
  TileLayer,
  Marker,
  Popup,
  useMap,
} from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import type { ProjectDetailsProps, OrganizationLocation } from "../hooks/types";

import markerIconPng from "leaflet/dist/images/marker-icon.png";
import markerIcon2x from "leaflet/dist/images/marker-icon-2x.png";
import markerShadow from "leaflet/dist/images/marker-shadow.png";

const customIcon = new L.Icon({
  iconUrl: markerIconPng,
  iconRetinaUrl: markerIcon2x,
  shadowUrl: markerShadow,
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
});

function ResizeMap({ count }: { count: number }) {
  const map = useMap();
  useEffect(() => {
    map?.invalidateSize();
  }, [count, map]);
  return null;
}

export default function ProjectDetails({
  project,}: ProjectDetailsProps) {
  // fetch organization locations
  const [orgLocations, setOrgLocations] = useState<OrganizationLocation[]>([]);
  const [loadingOrgs, setLoadingOrgs] = useState(true);
  const [loadingPlot, setLoadingPlot] = useState(true);

  useEffect(() => {
    if (!project) return;
    setLoadingOrgs(true);
    fetch(`/api/project/${project.id}/organizations`)
      .then((r) => r.json())
      .then((data) => Array.isArray(data) ? setOrgLocations(data) : console.error(data))
      .catch(console.error)
      .finally(() => setLoadingOrgs(false));
  }, [project]);

  if (!project) {
    return (
      <Box p={6} textAlign="center">
        <Text color="gray.500">No project selected.</Text>
      </Box>
    );
  }
  const shapData = project.explanations;
  const predicted = project.predicted_label;
  const probability = project.predicted_prob;
  // Map center fallback
  const center: [number, number] = orgLocations.length
    ? [orgLocations[0].latitude, orgLocations[0].longitude]
    : [51.505, -0.09];

  return (
    <Flex direction={{ base: "column", md: "row" }} gap={8}>
      {/* Left: Details */}
      <Box flex={1} p={6} bg="white" borderRadius="md" boxShadow="sm">
        <HStack mb={4} align="baseline">
          <Heading size="l">{project.title}</Heading>
          <Badge colorScheme={project.status === "CLOSED" ? "green" : "red"}>
            {project.status}
          </Badge>
        </HStack>

        <SimpleGrid columns={[1, 2]} spacing={4} mb={6}>
          <Box>
            <Text fontWeight="bold">ID</Text>
            <Text>{project.id}</Text>
          </Box>
          <Box>
            <Text fontWeight="bold">Acronym</Text>
            <Text>{project.acronym}</Text>
          </Box>
          <Box>
            <Text fontWeight="bold">Start Date</Text>
            <Text>{project.startDate}</Text>
          </Box>
          <Box>
            <Text fontWeight="bold">End Date</Text>
            <Text>{project.endDate}</Text>
          </Box>
          <Box>
            <Text fontWeight="bold">Funding (EC max)</Text>
            <Text>€{project.ecMaxContribution.toLocaleString()}</Text>
          </Box>
          <Box>
            <Text fontWeight="bold">Legal Basis</Text>
            <Text>{project.legalBasis}</Text>
          </Box>
          <Box gridColumn={["auto", "span 2"]}>
            <Text fontWeight="bold">Framework Programme</Text>
            <Text>{project.frameworkProgramme}</Text>
          </Box>
        </SimpleGrid>

        <Box mb={6}>
          <Heading size="md" mb={2}>Objective</Heading>
          <Text whiteSpace="pre-wrap">{project.objective}</Text>
        </Box>

        {(project.list_euroSciVocTitle ?? []).length > 0 && (
          <Box mb={4}>
            <Heading size="md" mb={2}>EuroSciVoc Titles</Heading>
            <Wrap>
              {(project.list_euroSciVocTitle ?? []).map((t) => (
                <Tag key={t} mb={2}>{t}</Tag>
              ))}
            </Wrap>
          </Box>
        )}

        {(project.list_euroSciVocPath ?? []).length > 0 && (
          <Box mb={4}>
            <Heading size="md" mb={2}>EuroSciVoc Paths</Heading>
            <Wrap>
              {(project.list_euroSciVocPath ?? []).map((p) => (
                <Tag key={p} mb={2}>{p}</Tag>
              ))}
            </Wrap>
          </Box>
        )}

        <Divider my={6} />

        <Heading size="md" mb={3}>Participating Organizations</Heading>
        {loadingOrgs ? (
          <Spinner />
        ) : (
          <Box w="100%" h="300px" borderRadius="md" overflow="hidden">
            <MapContainer center={center} zoom={4} style={{ height: '100%', width: '100%' }}>
              <ResizeMap count={orgLocations.length} />
              <TileLayer
                attribution="&copy; OpenStreetMap contributors"
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              />
              {orgLocations.map((org, i) => (
                <Marker key={i} position={[org.latitude, org.longitude]} icon={customIcon}>
                  <Popup>
                    <Text fontWeight="bold">{org.name}</Text>
                    <Text>{org.country}</Text>
                  </Popup>
                </Marker>
              ))}
            </MapContainer>
          </Box>
        )}
      </Box>

      {/* Right: Model Explanation */}
      <Box flex={{ base: '1', md: '0.6' }} bg="white" p={4} borderRadius="md" boxShadow="sm">
        <Heading size="sm" mb={4}>Model Prediction & Explanation</Heading>
        {shapData?.length ? (
          <>
            <Text mb={2}><strong>Predicted Label:</strong> {predicted}</Text>
            <Text mb={4}><strong>Probability:</strong> {(probability * 100).toFixed(2)}%</Text>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={shapData} margin={{ top: 10, right: 30, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="feature" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="shap" name="SHAP Value">
                  {shapData.map((entry, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={entry.shap >= 0 ? "#4caf50" : "#f44336"}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </>
        ) : (
          <Spinner />
        )}
      </Box>
    </Flex>
  );
}