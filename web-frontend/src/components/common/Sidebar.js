import React from "react";
import styled from "styled-components";
import {
  FiHome,
  FiTrendingUp,
  FiBriefcase,
  FiPieChart,
  FiSettings,
  FiLogOut,
} from "react-icons/fi";
import { Link, useLocation } from "react-router-dom";

const SidebarContainer = styled.aside`
  position: fixed;
  left: 0;
  top: 0;
  bottom: 0;
  width: 240px;
  background-color: ${(props) => props.theme.colors.backgroundLight};
  border-right: 1px solid ${(props) => props.theme.colors.border};
  padding-top: 70px;
  transition: transform 0.3s ease;
  z-index: 90;

  @media (max-width: ${(props) => props.theme.breakpoints.tablet}) {
    transform: translateX(${(props) => (props.isOpen ? "0" : "-100%")});
  }
`;

const Logo = styled.div`
  padding: 0 20px 20px;
  display: flex;
  align-items: center;
  justify-content: center;

  h1 {
    font-size: 24px;
    font-weight: 700;
    color: ${(props) => props.theme.colors.primary};
    margin: 0;
  }

  span {
    color: ${(props) => props.theme.colors.secondary};
  }
`;

const NavMenu = styled.nav`
  margin-top: 20px;
`;

const NavItem = styled(Link)`
  display: flex;
  align-items: center;
  padding: 12px 20px;
  color: ${(props) =>
    props.active
      ? props.theme.colors.primary
      : props.theme.colors.textSecondary};
  background-color: ${(props) =>
    props.active ? "rgba(41, 98, 255, 0.1)" : "transparent"};
  border-left: 3px solid
    ${(props) => (props.active ? props.theme.colors.primary : "transparent")};
  transition: all 0.2s ease;

  &:hover {
    color: ${(props) => props.theme.colors.primary};
    background-color: rgba(41, 98, 255, 0.05);
  }

  svg {
    margin-right: 12px;
    font-size: 18px;
  }
`;

const SidebarFooter = styled.div`
  position: absolute;
  bottom: 0;
  width: 100%;
  padding: 20px;
  border-top: 1px solid ${(props) => props.theme.colors.border};
`;

const FooterText = styled.p`
  font-size: 12px;
  color: ${(props) => props.theme.colors.textSecondary};
  text-align: center;
  margin-bottom: 10px;
`;

const Sidebar = ({ isOpen }) => {
  const location = useLocation();

  const navItems = [
    { icon: <FiHome />, label: "Dashboard", path: "/" },
    { icon: <FiTrendingUp />, label: "Trading", path: "/trading" },
    { icon: <FiBriefcase />, label: "Portfolio", path: "/portfolio" },
    { icon: <FiPieChart />, label: "Analytics", path: "/analytics" },
    { icon: <FiSettings />, label: "Settings", path: "/settings" },
  ];

  return (
    <SidebarContainer isOpen={isOpen}>
      <Logo>
        <h1>
          Option<span>ix</span>
        </h1>
      </Logo>

      <NavMenu>
        {navItems.map((item, index) => (
          <NavItem
            key={index}
            to={item.path}
            active={location.pathname === item.path ? 1 : 0}
          >
            {item.icon}
            {item.label}
          </NavItem>
        ))}
      </NavMenu>

      <SidebarFooter>
        <FooterText>Optionix v1.0.0</FooterText>
        <NavItem
          to="/logout"
          as="button"
          style={{
            width: "100%",
            border: "none",
            cursor: "pointer",
            textAlign: "left",
          }}
        >
          <FiLogOut />
          Logout
        </NavItem>
      </SidebarFooter>
    </SidebarContainer>
  );
};

export default Sidebar;
