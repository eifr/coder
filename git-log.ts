/** Recent commits read from the opened project's local Git reflog. */

export interface RecentCommit {
  /** Abbreviated commit hash, or empty for a COMMIT_EDITMSG fallback. */
  hash: string
  message: string
}

export const MAX_RECENT_COMMITS = 6
const MAX_RELOG_BYTES = 256 * 1024

/**
 * Parses `.git/logs/HEAD`. Reflog lines end with a tab and a message such as
 * `commit: feat: ...`; checkout and clone entries are ignored.
 */
export function parseGitLog(text: string): RecentCommit[] {
  const byHash = new Map<string, RecentCommit>()
  for (const line of text.split('\n')) {
    const tab = line.indexOf('\t')
    if (tab < 0) continue
    const rawMessage = line.slice(tab + 1).trim()
    const match = /^commit(?: \([^)]+\))?: (.+)$/.exec(rawMessage)
    if (!match) continue
    const [oldHash, newHash] = line.slice(0, tab).trim().split(' ')
    if (!newHash || /^0+$/.test(newHash) || newHash === oldHash) continue
    byHash.set(newHash, { hash: newHash.slice(0, 7), message: match[1].trim() })
  }
  return [...byHash.values()].slice(-MAX_RECENT_COMMITS).reverse()
}

export async function readRecentCommits(handle: FileSystemDirectoryHandle): Promise<RecentCommit[]> {
  try {
    const git = await handle.getDirectoryHandle('.git')
    try {
      const logs = await git.getDirectoryHandle('logs')
      const head = await logs.getFileHandle('HEAD')
      const file = await head.getFile()
      const sliced = file.size > MAX_RELOG_BYTES
      const tail = sliced ? file.slice(file.size - MAX_RELOG_BYTES) : file
      const text = await tail.text()
      return parseGitLog(sliced ? text.slice(text.indexOf('\n') + 1) : text)
    } catch {
      // Repositories with reflogs disabled still expose the last message.
    }
    try {
      const edit = await git.getFileHandle('COMMIT_EDITMSG')
      const file = await edit.getFile()
      const message = (await file.text()).trim().split('\n')[0]?.trim()
      if (message) return [{ hash: '', message }]
    } catch {
      // No usable local Git metadata.
    }
    return []
  } catch {
    // `.git` can be missing, inaccessible, or a worktree pointer file.
    return []
  }
}
