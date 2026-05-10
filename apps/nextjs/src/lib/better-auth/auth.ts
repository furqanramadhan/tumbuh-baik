import { betterAuth } from "better-auth";
import { nextCookies } from "better-auth/next-js";
import { mongodbAdapter } from "better-auth/adapters/mongodb";
import { magicLink, admin } from "better-auth/plugins";
import { getClient } from "./db";
import { GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET } from "../env";

const dbClient = (await getClient()).db();

export const auth = betterAuth({
  database: mongodbAdapter(dbClient),

  socialProviders: {
    google: {
      clientId: GOOGLE_CLIENT_ID,
      clientSecret: GOOGLE_CLIENT_SECRET,
    },
  },

  emailAndPassword: {
    enabled: true,
  },

  trustedOrigins: [
    "https://zonapetik.site",
    "https://backed-octagon-levitate.ngrok-free.dev",
    "http://localhost:3000",
  ],

  session: {
    cookieCache: {
      enabled: true,
    },
  },

  advanced: {
    defaultCookieAttributes: {
      sameSite: "none",
      secure: true,
      httpOnly: true,
    },
  },

  plugins: [
    magicLink({
      sendMagicLink: async ({ email, url }) => {
        console.log(`Send login link to ${email}: ${url}`);
      },
    }),
    admin(),
    nextCookies(),
  ],
});
