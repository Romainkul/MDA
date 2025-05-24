import type { Dispatch, SetStateAction } from "react";
export interface Project {
  id: string | number;
  title: string;
  status: string;
  startDate: string;
  endDate: string;
  ecMaxContribution: number;
  acronym: string;
  legalBasis: string;
  objective: string;
  frameworkProgramme: string;
  list_euroSciVocTitle: string[];
  list_euroSciVocPath: string[];
  explanations: Array<{ feature: string; shap: number }>;
  predicted_label: number;
  predicted_prob: number;
  totalCost: number | null | undefined;
  publications:{ [type: string]: number };
  fundingScheme: string;
}

export interface ProjectDetailsProps {
  project: Project;
//  question: string;
//  setQuestion: React.Dispatch<React.SetStateAction<string>>;
//  askChatbot: () => void;
//  chatHistory: ChatMessage[];
//  messagesEndRef: React.RefObject<HTMLDivElement>;
}


export interface DashboardProps {
  stats: { [key: string]: { labels: string[]; values: number[] } };
  filters: FilterState;
  setFilters: Dispatch<SetStateAction<FilterState>>;
  availableFilters: AvailableFilters;
  project: Project[];
}

export interface OrganizationLocation {
  name: string;
  country: string;
  latitude: number;
  longitude: number;
  sme: boolean;
  city: string;
  role: string;
  contribution: number;
  activityType: string;
  orgURL: string;
}

export interface FilterState {
  [key: string]: string;
  status: string;
  organization: string;
  country: string;
  legalBasis: string;
  minYear: string;
  maxYear: string;
  minFunding: string;
  maxFunding: string;
}


export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface AvailableFilters {
  [key: string]: string[];
  statuses: string[];
  organizations: string[];
  countries: string[];
  legalBases: string[];
}

export interface ProjectExplorerProps {
  projects: Project[];
  search: string;
  setSearch: (value: string) => void;
  statusFilter: string;
  setStatusFilter: (value: string) => void;
  legalFilter: string;
  setLegalFilter: (value: string) => void;
  orgFilter: string;
  setOrgFilter: (value: string) => void;
  countryFilter: string;
  setCountryFilter: (value: string) => void;
  fundingSchemeFilter: string;
  setFundingSchemeFilter: (value: string) => void;
  idFilter: string;
  setIdFilter: (value: string) => void;
  setSortField: (field: string) => void;
  sortField: string;
  setSortOrder : (order: "asc" | "desc") => void;
  sortOrder : "asc" | "desc";
  page: number;
  setPage: React.Dispatch<React.SetStateAction<number>>;
  setSelectedProject: (project: Project) => void;
  question: string;
  setQuestion: (q: string) => void;
  chatHistory: ChatMessage[];
  askChatbot: () => Promise<void>;
  loading: boolean;
  messagesEndRef: React.RefObject<HTMLDivElement>;
}