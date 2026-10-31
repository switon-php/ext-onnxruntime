<?php

declare(strict_types=1);

function surface(string $class): array
{
    $ref = new ReflectionClass($class);
    $methods = [];
    foreach ($ref->getMethods(ReflectionMethod::IS_PUBLIC) as $method) {
        if ($method->getDeclaringClass()->getName() !== $class) {
            continue;
        }
        $params = [];
        foreach ($method->getParameters() as $param) {
            $params[] = [
                'name' => $param->getName(),
                'type' => $param->hasType() ? (string) $param->getType() : null,
                'optional' => $param->isOptional(),
                'default' => $param->isDefaultValueAvailable() ? $param->getDefaultValue() : null,
                'hasDefault' => $param->isDefaultValueAvailable(),
            ];
        }
        $methods[$method->getName()] = [
            'static' => $method->isStatic(),
            'return' => $method->hasReturnType() ? (string) $method->getReturnType() : null,
            'params' => $params,
        ];
    }
    ksort($methods);
    $constants = [];
    foreach ($ref->getReflectionConstants() as $constant) {
        if ($constant->getDeclaringClass()->getName() !== $class || !$constant->isPublic()) {
            continue;
        }
        $value = $constant->getValue();
        if ($value instanceof BackedEnum) {
            $value = $value->value;
        }
        $constants[$constant->getName()] = $value;
    }
    ksort($constants);
    $cases = [];
    if ($ref->isEnum()) {
        foreach ((new ReflectionEnum($class))->getCases() as $case) {
            $cases[$case->getName()] = $case->getBackingValue();
        }
        ksort($cases);
    }
    return [
        'final' => $ref->isFinal(),
        'methods' => $methods,
        'constants' => $constants,
        'cases' => $cases,
    ];
}

function surface_namespace(): array
{
    $out = [];
    foreach (get_declared_classes() as $class) {
        if (str_starts_with($class, 'Switon\\OnnxRuntime\\')) {
            $out[$class] = surface($class);
        }
    }
    ksort($out);
    return $out;
}
