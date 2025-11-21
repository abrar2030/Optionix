import React from "react";
import styled from "styled-components";
import { FiGithub, FiTwitter, FiLinkedin } from "react-icons/fi";

const FooterContainer = styled.footer`
  background-color: ${(props) => props.theme.colors.backgroundLight};
  border-top: 1px solid ${(props) => props.theme.colors.border};
  padding: 20px;
  text-align: center;
`;

const FooterContent = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  max-width: 1200px;
  margin: 0 auto;
`;

const FooterLogo = styled.div`
  margin-bottom: 16px;

  h2 {
    font-size: 20px;
    font-weight: 700;
    color: ${(props) => props.theme.colors.primary};
    margin: 0;
  }

  span {
    color: ${(props) => props.theme.colors.secondary};
  }
`;

const FooterLinks = styled.div`
  display: flex;
  gap: 20px;
  margin-bottom: 16px;

  @media (max-width: ${(props) => props.theme.breakpoints.mobile}) {
    flex-direction: column;
    gap: 10px;
  }
`;

const FooterLink = styled.a`
  color: ${(props) => props.theme.colors.textSecondary};
  text-decoration: none;
  font-size: 14px;

  &:hover {
    color: ${(props) => props.theme.colors.primary};
  }
`;

const SocialLinks = styled.div`
  display: flex;
  gap: 16px;
  margin-bottom: 16px;
`;

const SocialLink = styled.a`
  color: ${(props) => props.theme.colors.textSecondary};
  font-size: 20px;

  &:hover {
    color: ${(props) => props.theme.colors.primary};
  }
`;

const Copyright = styled.p`
  color: ${(props) => props.theme.colors.textSecondary};
  font-size: 12px;
  margin: 0;
`;

const Footer = () => {
  return (
    <FooterContainer>
      <FooterContent>
        <FooterLogo>
          <h2>
            Option<span>ix</span>
          </h2>
        </FooterLogo>

        <FooterLinks>
          <FooterLink href="#">About</FooterLink>
          <FooterLink href="#">Documentation</FooterLink>
          <FooterLink href="#">API</FooterLink>
          <FooterLink href="#">Pricing</FooterLink>
          <FooterLink href="#">Contact</FooterLink>
        </FooterLinks>

        <SocialLinks>
          <SocialLink href="#" aria-label="GitHub">
            <FiGithub />
          </SocialLink>
          <SocialLink href="#" aria-label="Twitter">
            <FiTwitter />
          </SocialLink>
          <SocialLink href="#" aria-label="LinkedIn">
            <FiLinkedin />
          </SocialLink>
        </SocialLinks>

        <Copyright>
          © {new Date().getFullYear()} Optionix. All rights reserved.
        </Copyright>
      </FooterContent>
    </FooterContainer>
  );
};

export default Footer;
