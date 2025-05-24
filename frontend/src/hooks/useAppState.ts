import { useState, useEffect, useRef } from "react";
import { debounce } from "lodash";
import type {
  Project,
  OrganizationLocation,
  FilterState,
  ChatMessage,
  AvailableFilters,
} from "./types";

interface Stats {
  [key: string]: {
    labels: string[];
    values: number[];
  };
}

type SortOrder = "asc" | "desc";

export const useAppState = () => {
  // Projects state and pagination
  const [projects, setProjects] = useState<Project[]>([]);
  const [page, setPage] = useState<number>(0);

  // Search and filter states
  const [search, setSearch] = useState<string>("");
  const [statusFilter, setStatusFilter] = useState<string>("");
  const [legalFilter, setLegalFilter] = useState<string>("");
  const [orgFilter, setOrgFilter] = useState<string>("");
  const [countryFilter, setCountryFilter] = useState<string>("");
  const [fundingSchemeFilter, setFundingSchemeFilter] = useState<string>("");
  const [idFilter, setIdFilter] = useState<string>("");

  // Sorting
  const [sortField, setSortField] = useState<string>("");
  const [sortOrder, setSortOrder] = useState<SortOrder>("asc");

  // Dashboard stats and available filters
  const [stats, setStats] = useState<Stats>({});
  const [filters, setFilters] = useState<FilterState>({
    status: "",
    organization: "",
    country: "",
    legalBasis: "",
    minYear: "2000",
    maxYear: "2025",
    minFunding: "0",
    maxFunding: "10000000",
  });
  const [availableFilters, setAvailableFilters] = useState<AvailableFilters>({
    statuses: ["SIGNED", "CLOSED", "TERMINATED", "UNKNOWN"],
    organizations: [],
    countries: [],
    legalBases: [],
    fundingSchemes: [],
    ids: [],
  });

  // Chatbot states
  const [question, setQuestion] = useState<string>("");
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  // Fetch projects with current filters, pagination, sorting
  const fetchProjects = () => {
    const query = new URLSearchParams({
      page: page.toString(),
      search,
      status: statusFilter,
      legalBasis: legalFilter,
      organization: orgFilter,
      country: countryFilter,
      fundingScheme: fundingSchemeFilter,
      proj_id: idFilter,
      sortField,
      sortOrder,
    }).toString();

    fetch(`/api/projects?${query}`)
      .then((res) => res.json())
      .then((data: Project[]) => setProjects(data))
      .catch((err) => console.error("Error fetching projects:", err));
  };

  // Fetch stats with debouncing to limit requests
  const fetchStats = debounce((filterParams: FilterState) => {
    const query = new URLSearchParams(filterParams as any).toString();
    fetch(`/api/stats?${query}`)
      .then((res) => res.json())
      .then((data: Stats) => setStats(data))
      .catch((err) => console.error("Error fetching stats:", err));
  }, 500);

  // Fetch available filter options based on dataset and active filters
  const fetchAvailableFilters = (filterParams: FilterState) => {
    const query = new URLSearchParams(filterParams as any).toString();
    fetch(`/api/filters?${query}`)
      .then((res) => res.json())
      .then((data) => {
        setAvailableFilters({
          statuses: ["SIGNED", "CLOSED", "TERMINATED", "UNKNOWN"],
          organizations: data.organizations,
          countries: data.countries,
          legalBases: data.legalBases,
          fundingSchemes: data.fundingSchemes,
          ids: [],
        });
      })
      .catch((err) => console.error("Error fetching filters:", err));
  };

  interface RagResponse {
    answer: string;
    source_ids: string[];
  }

  // Handle chat submission
  const askChatbot = async () => {
    if (!question.trim() || loading) return;

    // Append user message
    setChatHistory((prev) => [...prev, { role: "user", content: question }]);
    setQuestion("");
    setLoading(true);

    // Add placeholder while generating
    setChatHistory((prev) => [...prev, { role: "assistant", content: "Generating answer..." }]);

    try {
      const response = await fetch("/api/rag", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: question }),
      });

      const text = await response.text();
      if (!response.ok) {
        let detail = text;
        try { detail = JSON.parse(text).detail; } catch {}
        throw new Error(detail);
      }

      const result: RagResponse = JSON.parse(text);
      const sources = result.source_ids.length ? result.source_ids.join(", ") : "none";
      const assistantContent = `${result.answer}\n\nSources: ${sources}`;

      // Replace placeholder with actual answer
      setChatHistory((prev) => [...prev.slice(0, -1), { role: "assistant", content: assistantContent }]);
    } catch (err: any) {
      // Replace placeholder with error message
      setChatHistory((prev) => [
        ...prev.slice(0, -1),
        { role: "assistant", content: `Error: ${err.message}` },
      ]);
    } finally {
      setLoading(false);
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  };

  // Effects: refetch when filters/sorting/pagination change
  useEffect(() => {
    fetchProjects();
  }, [page, search, statusFilter, legalFilter, orgFilter, countryFilter, fundingSchemeFilter, idFilter, sortField, sortOrder]);

  useEffect(() => {
    fetchStats(filters);
  }, [filters]);

  useEffect(() => {
    fetchAvailableFilters(filters);
  }, [filters]);

  return {
    selectedProject: projects[0] || null,
    dashboardProps: { stats, filters, setFilters, availableFilters },
    explorerProps: {
      projects,
      search,
      setSearch,
      statusFilter,
      setStatusFilter,
      legalFilter,
      setLegalFilter,
      orgFilter,
      setOrgFilter,
      countryFilter,
      setCountryFilter,
      fundingSchemeFilter,
      setFundingSchemeFilter,
      idFilter,
      setIdFilter,
      sortField,
      setSortField,
      sortOrder,
      setSortOrder,
      page,
      setPage,
      setSelectedProject: () => {},
      question,
      setQuestion,
      chatHistory,
      askChatbot,
      loading,
      messagesEndRef,
    },
    detailsProps: {
      project: projects[0]!,
      question,
      setQuestion,
      chatHistory,
      askChatbot,
      loading,
      messagesEndRef,
    },
  };
};
