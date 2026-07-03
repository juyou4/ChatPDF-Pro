/**
 * Provider Logo resource map.
 *
 * The source package currently does not include the image assets that older
 * Windows builds imported from src/assets/images/providers. Returning no logo
 * lets ProviderAvatar use its built-in initial-avatar fallback instead of
 * failing Vite's import analysis during startup.
 */

export const PROVIDER_LOGO_MAP: Record<string, string> = {}

/**
 * Get provider logo URL.
 * @param providerId - Provider id
 * @returns Logo image URL if configured, otherwise undefined.
 */
export function getProviderLogo(providerId: string): string | undefined {
  return PROVIDER_LOGO_MAP[providerId]
}
