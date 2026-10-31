<?php

declare(strict_types=1);

/**
 * Extension version from the git tag, without a leading v and without the commit id.
 * A tag v1.0.1 prints 1.0.1. A commit after that tag prints 1.0.1-3. No tag prints 0.1.0.
 */

$root = dirname(__DIR__);

function git(string $root, array $args): ?string
{
    $command = array_merge(['git', '-C', $root], $args);
    $spec = [1 => ['pipe', 'w'], 2 => ['pipe', 'w']];
    $proc = proc_open($command, $spec, $pipes, $root);
    if (!is_resource($proc)) {
        return null;
    }
    $text = stream_get_contents($pipes[1]);
    fclose($pipes[1]);
    fclose($pipes[2]);
    $status = proc_close($proc);
    if ($status !== 0) {
        return null;
    }
    $text = trim($text);
    return $text === '' ? null : $text;
}

function strip_tag(string $tag): string
{
    if (str_starts_with($tag, 'v') && isset($tag[1]) && ctype_digit($tag[1])) {
        return substr($tag, 1);
    }
    return $tag;
}

function drop_commit(string $described): string
{
    $abbrevPos = strrpos($described, '-');
    if ($abbrevPos === false) {
        return $described;
    }
    $abbrev = substr($described, $abbrevPos + 1);
    $head = substr($described, 0, $abbrevPos);
    if (!str_starts_with($abbrev, 'g')) {
        return $described;
    }
    $hex = substr($abbrev, 1);
    if ($hex === '' || !ctype_xdigit($hex)) {
        return $described;
    }
    $distancePos = strrpos($head, '-');
    if ($distancePos === false) {
        return $described;
    }
    $distance = substr($head, $distancePos + 1);
    $tag = substr($head, 0, $distancePos);
    if ($distance === '' || !ctype_digit($distance)) {
        return $described;
    }
    return $tag . '-' . $distance;
}

$exact = git($root, ['describe', '--tags', '--exact-match']);
if ($exact !== null) {
    echo strip_tag($exact), "\n";
    exit(0);
}
$described = git($root, ['describe', '--tags']);
if ($described !== null) {
    echo drop_commit(strip_tag($described)), "\n";
    exit(0);
}
echo "0.1.0\n";
